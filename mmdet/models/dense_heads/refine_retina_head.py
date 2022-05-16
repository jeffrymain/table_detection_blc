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


@HEADS.register_module()
class RefineRetinaHead(RetinaHead):

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
