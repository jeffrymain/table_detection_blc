# Copyright (c) OpenMMLab. All rights reserved.
import torch
import numpy as np
from ...datasets.pipelines.formatting import to_tensor

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import HEADS, build_head, build_roi_extractor, build_shared_head
from .base_roi_head import BaseRoIHead
from .test_mixins import BBoxTestMixin, LBoxTestMixin, MaskTestMixin

from mmdet.models.roi_heads import StandardRoIHead

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
class RefineStandardRoIHead(StandardRoIHead, LBoxTestMixin):
    """Simplest base roi head including one bbox head and one mask head."""
    def __init__(self,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 lbox_roi_extractor=None,
                 lbox_head=None,
                 shared_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(BaseRoIHead, self).__init__(init_cfg)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if shared_head is not None:
            shared_head.pretrained = pretrained
            self.shared_head = build_shared_head(shared_head)

        if bbox_head is not None:
            self.init_bbox_head(bbox_roi_extractor, bbox_head)

        if mask_head is not None:
            self.init_mask_head(mask_roi_extractor, mask_head)
        
        if lbox_head is not None:
            self.init_lbox_head(lbox_roi_extractor, lbox_head)

        self.init_assigner_sampler()
    
    def init_lbox_head(self, lbox_roi_extractor, lbox_head):
        """初始化线段分类头"""
        self.lbox_roi_extractor = build_roi_extractor(lbox_roi_extractor)
        self.lbox_head = build_head(lbox_head)

    # TODO 需要修改
    def forward_dummy(self, x, proposals):
        """Dummy forward function."""
        # bbox head
        outs = ()
        rois = bbox2roi([proposals])
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            outs = outs + (bbox_results['cls_score'],
                           bbox_results['bbox_pred'])
        # mask head
        if self.with_mask:
            mask_rois = rois[:100]
            mask_results = self._mask_forward(x, mask_rois)
            outs = outs + (mask_results['mask_pred'], )
        return outs

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas)
            losses.update(bbox_results['loss_bbox'])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            losses.update(mask_results['loss_mask'])
        
        device = x[0].device
        line_bboxes_list = [img_meta['line_bboxes'].data.to(device) for img_meta in img_metas]
        line_labels_list = [img_meta['line_labels'].data.type(torch.int64).to(device) for img_meta in img_metas]

        lbox_results = self._lbox_forward_train(x, line_bboxes_list, line_labels_list)
        losses.update(lbox_results['loss_lbox'])

        return losses
    
    def _lbox_forward(self, x, rois):
        lbox_feats = self.lbox_roi_extractor(
            x[:self.lbox_roi_extractor.num_inputs], rois
        )
        if self.with_shared_head:
            lbox_feats = self.lbox_head(lbox_feats)
        
        cls_score = self.lbox_head(lbox_feats)

        lbox_results = dict(
            cls_score=cls_score, lbox_feats=lbox_feats
        )
        return lbox_results

    def _lbox_forward_train(self, x, line_bboxes_list, line_labels_list):
        rois = bbox2roi(line_bboxes_list)
        lbox_results = self._lbox_forward(x, rois)

        # lbox_targets = self.lbox_head.get_targets(sampling_results, gt_bboxes,
        #                                           gt_labels, self.train_cfg)

        line_labels = torch.cat(line_labels_list, 0)

        label_weights = line_labels.new_ones(line_labels.size(0))
        loss_lbox = self.lbox_head.loss(
            lbox_results['cls_score'],
            line_labels,
            label_weights
        )
        lbox_results.update(loss_lbox=loss_lbox)
        return lbox_results

    async def async_simple_test(self,
                                x,
                                proposal_list,
                                img_metas,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = await self.async_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)
        if not self.with_mask:
            return bbox_results
        else:
            segm_results = await self.async_test_mask(
                x,
                img_metas,
                det_bboxes,
                det_labels,
                rescale=rescale,
                mask_test_cfg=self.test_cfg.get('mask'))
            return bbox_results, segm_results

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False):
        """Test without augmentation.

        Args:
            x (tuple[Tensor]): Features from upstream network. Each
                has shape (batch_size, c, h, w).
            proposal_list (list(Tensor)): Proposals from rpn head.
                Each has shape (num_proposals, 5), last dimension
                5 represent (x1, y1, x2, y2, score).
            img_metas (list[dict]): Meta information of images.
            rescale (bool): Whether to rescale the results to
                the original image. Default: True.

        Returns:
            list[list[np.ndarray]] or list[tuple]: When no mask branch,
            it is bbox results of each image and classes with type
            `list[list[np.ndarray]]`. The outer list
            corresponds to each image. The inner list
            corresponds to each class. When the model has mask branch,
            it contains bbox results and mask results.
            The outer list corresponds to each image, and first element
            of tuple is bbox results, second element is mask results.
        """
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        device = x[0].device
        ori_lines_list = [img_meta['ori_lines'] for img_meta in img_metas]
        line_bboxes_list = [to_tensor(img_meta['line_bboxes']).to(device) for img_meta in img_metas]
        det_lboxes, det_line_labels = self.simple_test_lboxes(
            x, img_metas, line_bboxes_list, self.test_cfg.cls_line_cfg, rescale=rescale
        )
        line_results = [
            line2result(ori_lines_list[i], det_lboxes[i], det_line_labels[i], self.lbox_head.num_classes)
            for i in range(len(det_line_labels))    # <---- batch长度
        ]


        if not self.with_mask:
            # return bbox_results
            # return [(img_bbox_results, img_line_results) for img_bbox_results, img_line_results in zip(bbox_results, line_results)]
            return list(zip(bbox_results, line_results))
        else:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return list(zip(bbox_results, segm_results, line_results))

    def aug_test(self, x, proposal_list, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        det_bboxes, det_labels = self.aug_test_bboxes(x, img_metas,
                                                      proposal_list,
                                                      self.test_cfg)
        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= det_bboxes.new_tensor(
                img_metas[0][0]['scale_factor'])
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_mask:
            segm_results = self.aug_test_mask(x, img_metas, det_bboxes,
                                              det_labels)
            return [(bbox_results, segm_results)]
        else:
            return [bbox_results]

