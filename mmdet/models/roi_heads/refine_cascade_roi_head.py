# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
from mmcv.runner import ModuleList

from mmdet.core import (bbox2result, bbox2roi, bbox_mapping, build_assigner,
                        build_sampler, merge_aug_bboxes, merge_aug_masks,
                        multiclass_nms)
from ..builder import HEADS, build_head, build_roi_extractor
from .base_roi_head import BaseRoIHead
from .test_mixins import BBoxTestMixin, MaskTestMixin

from .cascade_roi_head import CascadeRoIHead
import torch
import numpy as np
from torch import nn, Tensor, square, sqrt
from typing import Optional, List, Dict, Tuple


@HEADS.register_module()
class RefineCascadeRoIHead(CascadeRoIHead):


    def assign_line(
        self,
        pro_lines_per_img: Tensor,
        hough_lines_per_img
    ):
        out = pro_lines_per_img.unsqueeze(axis=1)
        out = out.repeat(1,2,1)

        list_pro_lines = list(pro_lines_per_img)
        # list_hough_lines = list(hough_lines_per_img)
        for i,pro_line in enumerate(pro_lines_per_img):
            diff = square(hough_lines_per_img - pro_line).sum(axis=1)
            min_idx = diff.argmin()
            if diff[min_idx] < 250:
                out[i,1,:] = hough_lines_per_img[min_idx]

        return out



    def comp_line(
        self, 
        proposals,  # type: List[Tensor]
        lines,
    ):

        dtype = proposals[0].dtype
        device = proposals[0].device
        # hough_lines = [t['lines'].to(dtype) for t in targets]
        hough_lines = []
        for i in range(len(lines)):
            if lines[i] is not None:
                hough_lines.append(lines[i].to(dtype).to(device))
        else:
            hough_lines.append(None)
        output = []

        for proposal_per_img, hough_lines_per_img in zip(proposals, hough_lines):
            
            proposal_per_img.unsqueeze_(dim=1)
            # 霍夫变换未检测到直线的情况下
            if hough_lines_per_img is None:
                out = proposal_per_img.repeat(1,2,1)
                output.append(out)
                continue

            pro_lines_per_img = proposal_per_img.repeat(1,4,1)
            pro_lines_per_img[:,0,3] = pro_lines_per_img[:,0,1]
            pro_lines_per_img[:,1,0] = pro_lines_per_img[:,1,2]
            pro_lines_per_img[:,2,1] = pro_lines_per_img[:,2,3]
            pro_lines_per_img[:,3,2] = pro_lines_per_img[:,3,0]
            pro_lines_per_img = pro_lines_per_img.reshape(-1,4)

            assing_lines = self.assign_line(pro_lines_per_img, hough_lines_per_img)
            (proposal_lines, refine_lines) = assing_lines.split(1,dim=1)
            proposal_lines = proposal_lines.reshape(-1,4,4)
            refine_lines = refine_lines.reshape(-1,4,4)
            proposals = proposal_lines[:,0,:].clone()
            proposals[:,3] = proposal_lines[:,2,3]
            refine_proposals = refine_lines[:,0,:].clone()
            refine_proposals[:,3] = refine_lines[:,2,3]

            out = torch.stack((proposals, refine_proposals),dim=1)
            output.append(out)

        return output
    # def assign_line(
    #     self,
    #     pro_lines_per_img: Tensor,
    #     hough_lines_per_img,
    #     pro_x: Tensor,
    #     pro_y: Tensor
    # ):

    #     out = pro_lines_per_img.unsqueeze(axis=1)
    #     out = out.repeat(1,2,1)

    #     list_pro_lines = list(pro_lines_per_img)
    #     # list_hough_lines = list(hough_lines_per_img)
    #     line_num = pro_lines_per_img.shape[0] - 1
    #     for i,pro_line in enumerate(pro_lines_per_img):
    #         # diff = square(hough_lines_per_img - pro_line).sum(axis=1)
    #         diff = square(hough_lines_per_img - pro_line)
    #         diff_1 = sqrt(diff[:,0] + diff[:,1])
    #         diff_2 = sqrt(diff[:,2] + diff[:,3]) 
    #         diff = (diff_1 + diff_2).div_(2.0)

    #         min_idx = diff.argmin()

    #         pro_i = int(line_num/4)
    #         if i % 2 == 0:  # 横线
    #             z = pro_y[pro_i] / 13.0
    #         else:
    #             z = pro_x[pro_i] / 13.0

    #         if diff[min_idx] < z:
    #             out[i,1,:] = hough_lines_per_img[min_idx]

    #     return out



    # def comp_line(
    #     self, 
    #     proposals,  # type: List[Tensor]
    #     lines,
    # ):

    #     dtype = proposals[0].dtype
    #     device = proposals[0].device
    #     # hough_lines = [t['lines'].to(dtype) for t in targets]
    #     hough_lines = []
    #     for i in range(len(lines)):
    #         if lines[i] is not None:
    #             hough_lines.append(lines[i].to(dtype).to(device))
    #     else:
    #         hough_lines.append(None)
    #     output = []

    #     for proposal_per_img, hough_lines_per_img in zip(proposals, hough_lines):
            
    #         pro_width = proposal_per_img[:,2] - proposal_per_img[:,0]
    #         pro_height = proposal_per_img[:,3] - proposal_per_img[:,1]

    #         proposal_per_img.unsqueeze_(dim=1)
    #         # 霍夫变换未检测到直线的情况下
    #         if hough_lines_per_img is None:
    #             out = proposal_per_img.repeat(1,2,1)
    #             output.append(out)
    #             continue

    #         pro_lines_per_img = proposal_per_img.repeat(1,4,1)
    #         pro_lines_per_img[:,0,3] = pro_lines_per_img[:,0,1]
    #         pro_lines_per_img[:,1,0] = pro_lines_per_img[:,1,2]
    #         pro_lines_per_img[:,2,1] = pro_lines_per_img[:,2,3]
    #         pro_lines_per_img[:,3,2] = pro_lines_per_img[:,3,0]
    #         pro_lines_per_img = pro_lines_per_img.reshape(-1,4)

    #         assing_lines = self.assign_line(pro_lines_per_img, hough_lines_per_img, pro_width, pro_height)
    #         (proposal_lines, refine_lines) = assing_lines.split(1,dim=1)
    #         proposal_lines = proposal_lines.reshape(-1,4,4)
    #         refine_lines = refine_lines.reshape(-1,4,4)
    #         proposals = proposal_lines[:,0,:].clone()
    #         proposals[:,3] = proposal_lines[:,2,3]
    #         refine_proposals = refine_lines[:,0,:].clone()
    #         refine_proposals[:,3] = refine_lines[:,2,3]

    #         out = torch.stack((proposals, refine_proposals),dim=1)
    #         output.append(out)

    #     return output

    def simple_test(self, x, proposal_list, img_metas, rescale=False):
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
        num_imgs = len(proposal_list)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        ori_shapes = tuple(meta['ori_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # "ms" in variable names means multi-stage
        ms_bbox_result = {}
        ms_segm_result = {}
        ms_scores = []
        rcnn_test_cfg = self.test_cfg

        rois = bbox2roi(proposal_list)

        if rois.shape[0] == 0:
            # There is no proposal in the whole batch
            bbox_results = [[
                np.zeros((0, 5), dtype=np.float32)
                for _ in range(self.bbox_head[-1].num_classes)
            ]] * num_imgs

            if self.with_mask:
                mask_classes = self.mask_head[-1].num_classes
                segm_results = [[[] for _ in range(mask_classes)]
                                for _ in range(num_imgs)]
                results = list(zip(bbox_results, segm_results))
            else:
                results = bbox_results

            return results

        for i in range(self.num_stages):
            bbox_results = self._bbox_forward(i, x, rois)

            # split batch bbox prediction back to each image
            cls_score = bbox_results['cls_score']
            bbox_pred = bbox_results['bbox_pred']
            num_proposals_per_img = tuple(
                len(proposals) for proposals in proposal_list)
            rois = rois.split(num_proposals_per_img, 0)
            cls_score = cls_score.split(num_proposals_per_img, 0)
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head[i].bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
            ms_scores.append(cls_score)

            if i < self.num_stages - 1:
                if self.bbox_head[i].custom_activation:
                    cls_score = [
                        self.bbox_head[i].loss_cls.get_activation(s)
                        for s in cls_score
                    ]
                refine_rois_list = []
                for j in range(num_imgs):
                    if rois[j].shape[0] > 0:
                        bbox_label = cls_score[j][:, :-1].argmax(dim=1)
                        refined_rois = self.bbox_head[i].regress_by_class(
                            rois[j], bbox_label, bbox_pred[j], img_metas[j])
                        refine_rois_list.append(refined_rois)
                rois = torch.cat(refine_rois_list)

        # average scores of each image by stages
        cls_score = [
            sum([score[i] for score in ms_scores]) / float(len(ms_scores))
            for i in range(num_imgs)
        ]

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(num_imgs):
            det_bbox, det_label = self.bbox_head[-1].get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        
        # 修正目标框
        if self.test_cfg.use_refine:
            lines = list(meta['lines'] for meta in img_metas)
            boxes = [det_bbox[:,:4] for det_bbox in det_bboxes]
            output = self.comp_line(boxes, lines)

            for i in range(len(det_bboxes)):
                det_bboxes[i][:,:4] = output[i][:,1,:]
        
        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head[-1].num_classes)
            for i in range(num_imgs)
        ]
        ms_bbox_result['ensemble'] = bbox_results

        if self.with_mask:
            if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
                mask_classes = self.mask_head[-1].num_classes
                segm_results = [[[] for _ in range(mask_classes)]
                                for _ in range(num_imgs)]
            else:
                if rescale and not isinstance(scale_factors[0], float):
                    scale_factors = [
                        torch.from_numpy(scale_factor).to(det_bboxes[0].device)
                        for scale_factor in scale_factors
                    ]
                _bboxes = [
                    det_bboxes[i][:, :4] *
                    scale_factors[i] if rescale else det_bboxes[i][:, :4]
                    for i in range(len(det_bboxes))
                ]
                mask_rois = bbox2roi(_bboxes)
                num_mask_rois_per_img = tuple(
                    _bbox.size(0) for _bbox in _bboxes)
                aug_masks = []
                for i in range(self.num_stages):
                    mask_results = self._mask_forward(i, x, mask_rois)
                    mask_pred = mask_results['mask_pred']
                    # split batch mask prediction back to each image
                    mask_pred = mask_pred.split(num_mask_rois_per_img, 0)
                    aug_masks.append([
                        m.sigmoid().cpu().detach().numpy() for m in mask_pred
                    ])

                # apply mask post-processing to each image individually
                segm_results = []
                for i in range(num_imgs):
                    if det_bboxes[i].shape[0] == 0:
                        segm_results.append(
                            [[]
                             for _ in range(self.mask_head[-1].num_classes)])
                    else:
                        aug_mask = [mask[i] for mask in aug_masks]
                        merged_masks = merge_aug_masks(
                            aug_mask, [[img_metas[i]]] * self.num_stages,
                            rcnn_test_cfg)
                        segm_result = self.mask_head[-1].get_seg_masks(
                            merged_masks, _bboxes[i], det_labels[i],
                            rcnn_test_cfg, ori_shapes[i], scale_factors[i],
                            rescale)
                        segm_results.append(segm_result)
            ms_segm_result['ensemble'] = segm_results

        if self.with_mask:
            results = list(
                zip(ms_bbox_result['ensemble'], ms_segm_result['ensemble']))
        else:
            results = ms_bbox_result['ensemble']

        return results

