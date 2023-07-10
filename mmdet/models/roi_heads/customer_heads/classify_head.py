import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from mmcv.runner import BaseModule, auto_fp16, force_fp32

from mmdet.core import build_bbox_coder
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.losses import accuracy
from mmdet.models.utils.builder import build_linear_layer


@HEADS.register_module()
class ClassifyHead(BaseModule):
    """用于分类的ROI HEAD，
    """
    def __init__(
        self,
        with_avg_pool=False,
        roi_feat_size=7,
        in_channels=256,
        num_classes=1,
        fc_out_channels=1024,
        fc1_cfg=dict(type='Linear'),
        fc2_cfg=dict(type='Linear'),
        loss_cls=dict(
            type="CrossEntropyLoss",
            use_sigmoid=False,
            loss_weight=1.0
        ),
        init_cfg=None
    ):
        super(ClassifyHead, self).__init__(init_cfg)
        self.with_avg_pool = with_avg_pool
        self.roi_feat_size = _pair(roi_feat_size)
        self.roi_feat_area = self.roi_feat_size[0] * self.roi_feat_size[1]
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fc_out_channels = fc_out_channels
        self.fc1_cfg = fc1_cfg
        self.fc2_cfg = fc2_cfg

        self.fp16_enabled = False

        self.loss_cls = build_loss(loss_cls)

        in_channels = self.in_channels
        if self.with_avg_pool:
            self.with_avg_pool = nn.AvgPool2d(self.roi_feat_size)
        else:
            in_channels *= self.roi_feat_area
        # 添加背景类别
        cls_channels = self.num_classes + 1

        self.fc1_cls = build_linear_layer(
            self.fc1_cfg,
            in_features=in_channels,
            out_features=self.fc_out_channels
        )
        self.fc2_cls = build_linear_layer(
            self.fc2_cfg,
            in_features=self.fc_out_channels,
            out_features=cls_channels
        )
        self.debug_imgs = None
        if init_cfg is None:
            self.init_cfg = []
            self.init_cfg += [
                dict(
                    type='Normal', std=0.01, override=dict(name ='fc1_cls')
                )
            ]
            self.init_cfg += [
                dict(
                    type='Normal', std=0.001, override=dict(name='fc2_cls')
                )
            ]
    
    @property
    def custom_activation(self):
        return getattr(self.loss_cls, 'custom_activation', False)
        
    @auto_fp16()
    def forward(self, x):
        if self.with_avg_pool:
            if x.numel() > 0:
                x = self.avg_pool(x)
                x = x.view(x.size(0), -1)
            else:
                # avg_pool does not support empty tensor,
                # so use torch.mean instead it
                x = torch.mean(x, dim=(-1, -2))
        x = x.flatten(1)
        middle_feature = self.fc1_cls(x)
        cls_score = self.fc2_cls(middle_feature)
        return cls_score
    
    @force_fp32(apply_to=('cls_score'))
    def loss(
        self,
        cls_score,
        labels,
        label_weights,
        reduction_override=None
    ):
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                loss_cls_ = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override
                )
            if isinstance(loss_cls_, dict):
                losses.update(loss_cls_)
            else:
                losses['line_loss_cls'] = loss_cls_
            if self.custom_activation:
                acc_ = self.loss_cls.get_accuracy(cls_score, labels)
                losses.update(acc_)
            else:
                losses['line_acc'] = accuracy(cls_score, labels)
        
        return losses

    @force_fp32(apply_to=('cls_score'))
    def get_line_labels(self, cls_score, lbox_pred, img_shape, scale_factor, rescale=False, cfg=None):
        multi_scores = F.softmax(cls_score, dim=-1) if cls_score is not None else None

        if lbox_pred is not None:
            lboxes = lbox_pred

        if rescale and lboxes.size(0) > 0:
            scale_factor = lboxes.new_tensor(scale_factor)
            lboxes = (lboxes.view(lboxes.size(0), -1, 4) / scale_factor).view(
                lboxes.size()[0], -1
            )

        if cfg is None:
            return lboxes, multi_scores
        else:
            num_classes = multi_scores.size(1)

            # TODO: 这里的实现是超过概率一半判为正类
            scores, labels = torch.max(multi_scores, dim=1)
            det_lboxes = torch.empty(lboxes.shape[0], 5)
            det_lboxes[:,:4] = lboxes[:]
            det_lboxes[:,4] = scores
            for i, d in enumerate(labels):
                if d != 0:
                    if det_lboxes[i, 4] <= 0.95:
                        det_lboxes[i, 4] = multi_scores[i, 0]
                        labels[i] = 0

            # scores = multi_scores[:, :-1]
            # labels = torch.arange(num_classes, dtype=torch.long, device=scores.device)
            # labels = labels.view(1, -1).expand_as(scores)
            # scores = scores.reshape(-1)
            # labels = labels.reshape(-1)
            # valid_mask = scores > cfg.pos_thr
            # inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
            # scores, lables = scores[inds], labels[inds]
            return det_lboxes, labels

            
