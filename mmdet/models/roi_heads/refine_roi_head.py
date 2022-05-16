# Copyright (c) OpenMMLab. All rights reserved.
from tkinter import N
from mmdet.core import bbox2result
from mmdet.models.builder import HEADS
from mmdet.models.roi_heads import StandardRoIHead
import torch
import numpy as np
from torch import nn, Tensor, square, sqrt
from typing import Optional, List, Dict, Tuple


@HEADS.register_module()
class RefineRoIHead(StandardRoIHead):
    """RoI head for 

    """

    def assign_line(
        self,
        pro_lines_per_img: Tensor,
        hough_lines_per_img,
        pro_x: Tensor,
        pro_y: Tensor
    ):

        out = pro_lines_per_img.unsqueeze(axis=1)
        out = out.repeat(1,2,1)

        list_pro_lines = list(pro_lines_per_img)
        # list_hough_lines = list(hough_lines_per_img)
        line_num = pro_lines_per_img.shape[0] - 1
        for i,pro_line in enumerate(pro_lines_per_img):
            # diff = square(hough_lines_per_img - pro_line).sum(axis=1)
            diff = square(hough_lines_per_img - pro_line)
            diff_1 = sqrt(diff[:,0] + diff[:,1])
            diff_2 = sqrt(diff[:,2] + diff[:,3]) 
            diff = (diff_1 + diff_2).div_(2.0)

            min_idx = diff.argmin()

            pro_i = int(line_num/4)
            if i % 2 == 0:  # 横线
                z = pro_y[pro_i] / 5.0
            else:
                z = pro_x[pro_i] / 5.0

            if diff[min_idx] < z:
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
            
            pro_width = proposal_per_img[:,2] - proposal_per_img[:,0]
            pro_height = proposal_per_img[:,3] - proposal_per_img[:,1]

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

            assing_lines = self.assign_line(pro_lines_per_img, hough_lines_per_img, pro_width, pro_height)
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

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    lines=None,
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

        # 修正目标框
        if self.test_cfg.use_refine:
            lines = list(meta['lines'] for meta in img_metas)
            boxes = [det_bbox[:,:4] for det_bbox in det_bboxes]
            output = self.comp_line(boxes, lines)
            # a = output[0][:,0,:]
            # b = output[0][:,1,:]
            # if not a.equal(b):
            #     print(img_metas[0]['ori_filename'])

            for i in range(len(det_bboxes)):
                det_bboxes[i][:,:4] = output[i][:,1,:]

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]
        
    
        # for bbox_result in bbox_results

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return list(zip(bbox_results, segm_results))
