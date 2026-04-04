from .transformer import *
from .losses import ctr_diou_loss_1d, sigmoid_focal_loss, kl_divergence_loss
from .softnms import soft_nms_intervals_cpu
import numpy as np


class MMCTransformer(nn.Module):
    def __init__(self, vis_dim, aud_dim, text_dim, d_model, self_num_layers, text_num_layers, cross_num_layers, num_heads, d_ff=2048):
        super(MMCTransformer, self).__init__()
        # UniModal Encoders for each modality
        self.vis_encoder = UniModalEncoder(vis_dim, d_model, self_num_layers, num_heads, d_ff)
        self.aud_encoder = UniModalEncoder(aud_dim, d_model, self_num_layers, num_heads, d_ff)
        self.text_encoder = UniModalEncoder(text_dim, d_model, self_num_layers, num_heads, d_ff)

        self.vis_text_croos_att = nn.ModuleList([CrossSelfEncoderLayer(d_model, num_heads, d_ff) for _ in range(text_num_layers)])
        self.aud_text_croos_att = nn.ModuleList([CrossSelfEncoderLayer(d_model, num_heads, d_ff) for _ in range(text_num_layers)])

        self.vis_aud_cross_att = nn.ModuleList([CrossAttentionEncoderLayer(d_model, num_heads, d_ff) for _ in range(cross_num_layers)])
        self.aud_vis_cross_att = nn.ModuleList([CrossAttentionEncoderLayer(d_model, num_heads, d_ff) for _ in range(cross_num_layers)])

        hidden_dim = 256
        
        self.feature_map = nn.Linear(d_model*2, d_model)

        self.cls_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

        # Unimodal cls heads for Multi-Modal Alignment Guider (KL divergence alignment)
        self.vis_cls_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

        self.aud_cls_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

        self.reg_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, 2),
            nn.ReLU()
        )

    def forward(self, batch):
        vis_feats = batch['visual_feats']
        aud_feats = batch['audio_feats']
        text_feats = batch['text_feats']
        masks = batch['masks']
        gt_cls_labels = batch['labels']
        gt_offsets = batch['segments']
        # preprocessing 
        # vis: [batch_size, seq_len, vis_dim]
        # aud: [batch_size, seq_len, aud_dim]
        # masks: [batch_size, 1, max_len]

        # encode the features via self-attention
        vis_feats = self.vis_encoder(vis_feats, masks)
        aud_feats = self.aud_encoder(aud_feats, masks)
        text_feats = self.text_encoder(text_feats, masks)

        # cross attention between vis and text
        for idx, layer in enumerate(self.vis_text_croos_att):
            vis_feats = layer(vis_feats, text_feats, masks)
        
        # cross attention between aud and text
        for idx, layer in enumerate(self.aud_text_croos_att):
            aud_feats = layer(aud_feats, text_feats, masks)


        for idx, layer in enumerate(self.vis_aud_cross_att):
            vis_feats = layer(vis_feats, aud_feats, masks)

        for idx, layer in enumerate(self.aud_vis_cross_att):
            aud_feats = layer(aud_feats, vis_feats, masks)


        # concat them together
        feats = torch.cat((vis_feats, aud_feats), dim=2)

        feats = self.feature_map(feats)

        # out_cls: [B, seq_len, 1] — multi-modal branch
        out_cls_logits = self.cls_head(feats)
        # unimodal branches for alignment guider
        out_vis_cls_logits = self.vis_cls_head(vis_feats)
        out_aud_cls_logits = self.aud_cls_head(aud_feats)
        # out_offset: [B, seq_len, 2]
        out_offsets = self.reg_head(feats)

        return masks, out_cls_logits, out_vis_cls_logits, out_aud_cls_logits, out_offsets, gt_cls_labels, gt_offsets

    @property
    def device(self):
        # a hacky way to get the device type
        # will throw an error if parameters are on different devices
        return list(set(p.device for p in self.parameters()))[0]


    def losses(
        self, masks,
        out_cls_logits, out_vis_cls_logits, out_aud_cls_logits,
        out_offsets,
        gt_cls_labels, gt_offsets
    ):
        # masks: [B, 1, max_len]
        # out_cls_logits / out_vis_cls_logits / out_aud_cls_logits: [B, seq_len, 1]
        # out_offsets: [B, seq_len, 2]
        # gt_cls_labels: [B, seq_len]
        # gt_offsets: [B, seq_len, 2]

        gt_cls_labels = gt_cls_labels.unsqueeze(-1)           # [B, T, 1]
        masks = masks.transpose(1, 2).contiguous()            # [B, T, 1]

        # 1. Uni-modal focal loss: L_focal^uni = focal(vis) + focal(aud)
        vis_focal = sigmoid_focal_loss(out_vis_cls_logits, gt_cls_labels)
        aud_focal = sigmoid_focal_loss(out_aud_cls_logits, gt_cls_labels)
        uni_focal_loss = ((vis_focal + aud_focal) * masks).sum()

        # 2. Multi-modal focal loss: L_focal^mul = focal(multi)
        mul_focal = sigmoid_focal_loss(out_cls_logits, gt_cls_labels)
        mul_focal_loss = (mul_focal * masks).sum()

        # 3. KL divergence alignment: L_KL^{v->m} + L_KL^{a->m}
        kl_v2m = kl_divergence_loss(out_vis_cls_logits, out_cls_logits, masks)
        kl_a2m = kl_divergence_loss(out_aud_cls_logits, out_cls_logits, masks)
        kl_loss = kl_v2m + kl_a2m

        # 4. Regression loss (IoU): only on positive positions
        cls_mask = (gt_cls_labels != 0).float()
        combined_mask = masks * cls_mask
        reg_loss = ctr_diou_loss_1d(out_offsets, gt_offsets)
        reg_loss = (reg_loss * combined_mask.squeeze(-1)).sum()

        return {
            'uni_focal_loss': uni_focal_loss,
            'mul_focal_loss': mul_focal_loss,
            'kl_loss'       : kl_loss,
            'reg_loss'      : reg_loss,
        }


    @torch.no_grad()
    def inference_single_video(self, masks, out_cls_logits, out_offsets, inference_settings):
        segs_all = []
        scores_all = []
        cls_idxs_all = []

        # sigmoid normalization for output logits
        pred_prob = (out_cls_logits.sigmoid().squeeze() * masks).flatten()

        # Apply filtering to make NMS faster
        # 1. Keep seg with confidence score > a threshold
        keep_idxs = (pred_prob > inference_settings['pre_nms_thresh'])

        pred_prob = pred_prob[keep_idxs]

        topk_idxs = keep_idxs.nonzero(as_tuple=True)[0]

        # 2. Keep top k top scoring boxes only
        num_topk = min(inference_settings['pre_nms_topk'], topk_idxs.size(0))
        pred_prob, idxs = pred_prob.sort(descending=True)
        pred_prob = pred_prob[:num_topk].clone()
        topk_idxs = topk_idxs[idxs[:num_topk]].clone()
        offsets = out_offsets[topk_idxs]

        # 3. compute predicted segments
        seg_left = topk_idxs - offsets[:, 0]
        seg_right = topk_idxs + offsets[:, 1]
        pred_segs = torch.stack((seg_left, seg_right), -1)

        # 4. Keep seg with duration > a threshold
        seg_durations = seg_right - seg_left
        keep_idxs2 = seg_durations > inference_settings['duration_thresh']
        keep_idxs3 = seg_durations < inference_settings['duration_thresh_max']

        keep_idxs2 = keep_idxs2 & keep_idxs3

        # *_all : N (filtered # of segments) x 2 / 1
        segs_all.append(pred_segs[keep_idxs2])
        scores_all.append(pred_prob[keep_idxs2])
        cls_idxs_all.append(topk_idxs[keep_idxs2])

        # cat along the seq_len
        segs_all, scores_all, cls_idxs_all = [
            torch.cat(x) for x in [segs_all, scores_all, cls_idxs_all]
        ]
        results = {'segments': segs_all,
                'scores': scores_all,
                'labels': cls_idxs_all}
        return results


    @torch.no_grad()
    def inference_(self, batch, inference_settings):
        
        masks, out_cls_logits, out_vis_cls_logits, out_aud_cls_logits, out_offsets, gt_cls_labels, gt_offsets = self.forward(batch)

        # batch seq_len
        pred_prob = out_cls_logits.squeeze(-1)

        results = []

        # 1: gather video meta information
        vid_idxs = batch['video_id']
        vid_lens = batch['duration']

        # 2: inference on each single video and gather the results
        # upto this point, all results use timestamps defined on feature grids
        for idx, (vidx, vlen) in enumerate(
            zip(vid_idxs, vid_lens)
        ):
            # gather per-video outputs
            cls_logits_per_vid = pred_prob[idx]
            offsets_per_vid = out_offsets[idx]
            masks_per_vid = masks[idx]
            mins = vlen // 60 
            max_seg_num = mins * inference_settings['max_seg_per_min']
            max_seg_num = int(np.ceil(max_seg_num))

            # inference on a single video (should always be the case)
            results_per_vid = self.inference_single_video(
                masks_per_vid,
                cls_logits_per_vid, offsets_per_vid, inference_settings
            )
            # results_per_vid_nms_idx = soft_nms_intervals(results_per_vid['scores'], results_per_vid['segments'], sigma=inference_settings['nms_sigma'], thresh=inference_settings['min_score'])
            results_per_vid_nms_idx = soft_nms_intervals_cpu(results_per_vid['scores'], results_per_vid['segments'], sigma=inference_settings['nms_sigma'], thresh=inference_settings['min_score'], max_seg_num=max_seg_num)
            results_per_vid['segments'] = results_per_vid['segments'][results_per_vid_nms_idx]
            results_per_vid['scores'] = results_per_vid['scores'][results_per_vid_nms_idx]
            results_per_vid['labels'] = results_per_vid['labels'][results_per_vid_nms_idx]
            # pass through video meta info
            results_per_vid['video_id'] = vidx
            results_per_vid['duration'] = vlen
            results.append(results_per_vid)

        return results