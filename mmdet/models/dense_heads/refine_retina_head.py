# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import ConvModule

from ..builder import HEADS
from .anchor_head import AnchorHead


from .retina_head import RetinaHead
import torch
import numpy as np
from torch import nn, Tensor, square
from typing import Optional, List, Dict, Tuple
from ..builder import HEADS, build_head, build_roi_extractor, build_shared_head

def line2result(ori_lines, lboxes, labels, num_classes):
    if lboxes.shape[0] == 0:
        return [np.zeros((0, 5), dtype=np.float32) for i in range(num_classes)]
    else:
        if isinstance(lboxes, torch.Tensor):
            # ori_lines = ori_lines.detach().cpu().numpy()
            lboxes = lboxes.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
    return [(ori_lines[labels == i, :], lboxes[labels == i, :]) for i in range(num_classes+1)]


@HEADS.register_module()
class RefineRetinaHead(RetinaHead):

    def __init__(
        self,
        num_classes,
        in_channels,
        stacked_convs=4,
        conv_cfg=None,
        norm_cfg=None,
        anchor_generator=dict(
        type='AnchorGenerator',
        octave_base_scale=4,
        scales_per_octave=3,
        ratios=[0.5, 1.0, 2.0],
        strides=[8, 16, 32, 64, 128]),
        init_cfg=dict(
        type='Normal',
        layer='Conv2d',
        std=0.01,
        override=dict(
            type='Normal',
            name='retina_cls',
            std=0.01,
            bias_prob=0.01)),
        lbox_roi_extractor=None,
        lbox_head=None,
        **kwargs
    ):
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        super(RetinaHead, self).__init__(
            num_classes,
            in_channels,
            anchor_generator=anchor_generator,
            init_cfg=init_cfg,
            **kwargs)

        if lbox_head is not None:
            self.init_lbox_head(lbox_roi_extractor, lbox_head)

    def init_lbox_head(self, lbox_roi_extractor, lbox_head):
        """初始化线段分类头"""
        self.lbox_roi_extractor = build_roi_extractor(lbox_roi_extractor)
        self.lbox_head = build_head(lbox_head)
    
    def simple_test_bboxes(self, feats, img_metas, rescale=False):
        """Test det bboxes without test-time augmentation, can be applied in
        DenseHead except for ``RPNHead`` and its variants, e.g., ``GARPNHead``,
        etc.

        Args:
            feats (tuple[torch.Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is ``bboxes`` with shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
                The shape of the second tensor in the tuple is ``labels``
                with shape (n,)
        """
        outs = self.forward(feats)
        results_list = self.get_bboxes(
            *outs, img_metas=img_metas, rescale=rescale)
        
        # 修正目标框
        # print(results_list[0][1].numel())
        # results_list = list(tuple(tensor, tensor))
        if self.test_cfg.use_refine:
            lines = list(meta['lines'] for meta in img_metas)

            results_list = [list(result) for result in results_list]

            boxes = [result[0][:,:4] for result in results_list]
            output = self.comp_line(boxes, lines)

            for i in range(len(results_list)):
                results_list[i][0][:,:4] = output[i][:,1,:]

            results_list = [tuple(result) for result in results_list]

        return results_list


    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        outs = self(x)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(
                *outs, img_metas=img_metas, cfg=proposal_cfg)
            return losses, proposal_list
