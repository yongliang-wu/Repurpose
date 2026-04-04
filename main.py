from dataset.RepurposeClip import RepurposeClip, RepurposeClipTest
from dataset.RepurposeClip import collate_fn, collate_fn_test
from models.MMCTransformer import MMCTransformer
from .utils.metrics import *
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from tqdm import tqdm
import numpy as np
import random
import time
import yaml
import os

def load_config(config_file):
    with open(config_file, "r") as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)
    return config

def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def main(args):
    # load config
    cfg = load_config(args.config_path)

    seed_everything(cfg['train']['seed'])

    checkpoint_path = time.strftime('saved_model/ckpt_%Y%m%d_%H%M%S')
    os.makedirs(checkpoint_path, exist_ok=True)

    with open(os.path.join(checkpoint_path, 'config.yaml'), 'w') as f:
        yaml.dump(cfg, f)

    print('The checkpoint path is %s' % checkpoint_path)

    train_dataset = RepurposeClip(**cfg['train_dataset'])
    test_dataset = RepurposeClipTest(**cfg['test_dataset'])

    model = MMCTransformer(**cfg['model']).to(device='cuda')

    learning_rate = cfg['train']['lr']
    weight_decay = cfg['train']['weight_decay']
    batch_size = cfg['train']['batch_size']
    num_epochs = cfg['train']['epochs']
    warmup_epochs = cfg['train']['warmup_epochs']

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=24)

    test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn_test, num_workers=24)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    num_iters = len(train_data_loader)

    warmup_steps = warmup_epochs * num_iters

    total_steps = num_epochs * num_iters

    def warmup_lambda(global_step):
        return (global_step + 1) / warmup_steps if (global_step + 1) <= warmup_steps else 1

    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)

    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    start_epoch = 0

    global_step = 0

    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        warmup_scheduler.load_state_dict(checkpoint['warmup_scheduler'])
        cosine_scheduler.load_state_dict(checkpoint['cosine_scheduler'])
        start_epoch = checkpoint['epoch']
        global_step = start_epoch * num_iters

    writer = SummaryWriter()
    best_tIoU = 0
    best_epoch = 0

    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0
        total_cls_loss = 0
        total_reg_loss = 0
        start_time = time.time()

        for i, batch in enumerate(train_data_loader):
            batch['visual_feats'] = batch['visual_feats'].to(device='cuda')
            batch['audio_feats'] = batch['audio_feats'].to(device='cuda')
            batch['text_feats'] = batch['text_feats'].to(device='cuda')
            batch['masks'] = batch['masks'].to(device='cuda')
            batch['labels'] = batch['labels'].to(device='cuda')
            batch['segments'] = batch['segments'].to(device='cuda')
            
            output = model(batch)
            losses = model.losses(*output)

            # Loss weights from paper: λ1=0.1, λ2=0.3, λ3=0.1, λ4=0.7
            # total = λ1*L_focal_uni + λ2*L_focal_mul + λ3*(L_KL_v2m + L_KL_a2m) + λ4*L_iou
            final_loss = (
                0.1 * losses['uni_focal_loss']
                + 0.3 * losses['mul_focal_loss']
                + 0.1 * losses['kl_loss']
                + 0.7 * losses['reg_loss']
            ) / batch_size
            optimizer.zero_grad()
            final_loss.backward()
            optimizer.step()

            uni_focal_loss = losses["uni_focal_loss"].item() / batch_size
            mul_focal_loss = losses["mul_focal_loss"].item() / batch_size
            kl_loss        = losses["kl_loss"].item() / batch_size
            reg_loss       = losses["reg_loss"].item() / batch_size
            batch_loss     = final_loss.item()
            total_cls_loss += mul_focal_loss
            total_reg_loss += reg_loss
            total_loss     += batch_loss

            step = epoch * len(train_data_loader) + i
            writer.add_scalar('Loss/uni_focal_loss', uni_focal_loss, step)
            writer.add_scalar('Loss/mul_focal_loss', mul_focal_loss, step)
            writer.add_scalar('Loss/kl_loss',        kl_loss,        step)
            writer.add_scalar('Loss/reg_loss',        reg_loss,       step)
            writer.add_scalar('Loss/total_loss',      batch_loss,     step)

            if global_step < warmup_steps:
                warmup_scheduler.step()
            else:
                cosine_scheduler.step()
            print(f"Epoch {epoch+1}/{num_epochs}, Iter {i+1}/{len(train_data_loader)}, Total: {batch_loss:.3f}, uni_focal: {uni_focal_loss:.3f}, mul_focal: {mul_focal_loss:.3f}, kl: {kl_loss:.3f}, reg: {reg_loss:.3f}, Time: {time.time() - start_time:.3f}s", end='\r')
        # save checkpoint to disk
        if epoch % cfg['train']['save_epochs'] == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'cosine_scheduler': cosine_scheduler.state_dict(),
                'warmup_scheduler': warmup_scheduler.state_dict(),
                'epoch': epoch,
                'loss': total_loss / len(train_data_loader)
            }
            torch.save(checkpoint, os.path.join(checkpoint_path, f'epoch_{epoch}.pth'))
        end_time = time.time()

        avg_cls_loss = total_cls_loss / len(train_data_loader)
        avg_reg_loss = total_reg_loss / len(train_data_loader)
        avg_total_loss = total_loss / len(train_data_loader)

        writer.add_scalar('Loss/avg_cls_loss', avg_cls_loss, epoch)
        writer.add_scalar('Loss/avg_reg_loss', avg_reg_loss, epoch)
        writer.add_scalar('Loss/avg_total_loss', avg_total_loss, epoch)

        epoch_message = f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_total_loss:.3f}, Avg cls Loss: {avg_cls_loss:.3f}, Avg reg Loss: {avg_reg_loss:.3f}, Time: {end_time - start_time:.3f}s"

        print(epoch_message)

        with open(os.path.join(checkpoint_path, 'a-log.txt'), 'a') as f:
            f.write(epoch_message + '\n')

        if epoch % cfg['train']['eval_freq'] == 0:
            model.eval()
            with torch.no_grad():
                total_AP = []
                totol_recall = []
                count = 0
                total_tIoU = []
                for batch in tqdm(test_data_loader):
                    count += 1
                    batch['visual_feats'] = batch['visual_feats'].to(device='cuda')
                    batch['audio_feats'] = batch['audio_feats'].to(device='cuda')
                    batch['text_feats'] = batch['text_feats'].to(device='cuda')
                    batch['masks'] = batch['masks'].to(device='cuda')
                    preds = model.inference_(batch, cfg['test_cfg'])
                    for i in range(len(preds)):
                        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
                        precision_per_threshold = calculate_tiou(batch['gt_segments'][i], preds[i]['segments'].tolist(), thresholds)
                        total_tIoU.append(precision_per_threshold)
                tIoU = {}
                for threshold in thresholds:
                    tIoU[threshold] = sum([item[threshold] for item in total_tIoU]) / len(total_tIoU)
                AtIoU = sum(item for item in tIoU.values()) / len(tIoU)

                if AtIoU > best_tIoU:
                    best_tIoU = AtIoU
                    best_epoch = epoch
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'cosine_scheduler': cosine_scheduler.state_dict(),
                        'warmup_scheduler': warmup_scheduler.state_dict(),
                        'epoch': epoch,
                        'loss': total_loss / len(train_data_loader)
                    }
                    torch.save(checkpoint, os.path.join(checkpoint_path, f'best.pth'))
                writer.add_scalar(f"Eval/AP_threshold={cfg['test_cfg']['pre_nms_thresh']}@tIoU", AtIoU, epoch)
                eval_message = f"Epoch {epoch+1}/{num_epochs}, tIoU: {tIoU}, best tIoU: {best_tIoU:.4f}, best epoch: {best_epoch}"
                print(eval_message)
                with open(os.path.join(checkpoint_path, 'log.txt'), 'a') as f:
                    f.write(eval_message + '\n')
    writer.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()
    main(args)