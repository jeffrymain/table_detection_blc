from ctypes import cdll
import enum
from errno import EDEADLK
import os.path as osp
from unittest import result
from xml.sax.handler import feature_namespace_prefixes
from pandas import wide_to_long

import os
import torch
import mmcv
import numpy as np
import pycocotools.mask as maskUtils

from mmdet.core import BitmapMasks, PolygonMasks, bbox
from ..builder import PIPELINES


from mmcv.parallel import DataContainer as DC
from .formatting import Collect, DefaultFormatBundle, to_tensor
from .transforms import Resize, RandomFlip

try:
    from panopticapi.utils import rgb2id
except ImportError:
    rgb2id = None

import cv2
import json
import ctypes

@PIPELINES.register_module()
class LoadLines:
    """利用霍夫变换载入图片中的直线
    HoughLinesP(image, rho, theta, threshold, line=None, minLineLength=None, maxLineGap=None)
        image, 二值图像
        rho, 线段以像素为单位的距离精度，double类型，推荐1.0
        theta, 线段以弧度为单位的角度精度，推荐用numpy.pi/180
        threshold, 累加平面的阈值参数，int类型，超过设定阈值才检测出线段，值越大，基本意味着检出的线段越长，推荐先使用100试试
        line, 意义未知 
        minLineLength, 线段以像素为单位的最小长度
        maxLineGap, 同一方向上的两条线段判定为一条线段的最大允许间隔（断裂），超过设定值，则把两条
    """
    def __init__(self, maxLineGap=5, pre_detected=True):
        self.maxLineGap = maxLineGap
        self.canny_lines = {}
        # canny line 的结果
        with open(r"D:\jeffry\Repos\sources\mmdetection\data\material_papers_e\ann\cannylines_result.json", 'r') as f:
            self.canny_lines = json.load(f)["line_detected"]
        pass

    def merge_line(self, lines):
        if lines.shape[0] == 1:
            return lines
        mask = [False] * lines.shape[0]

        merged_lines = []
        for i, line in enumerate(lines):
            if i >= lines.shape[0]-1:
                break
            if mask[i] is True:
                continue

            idx_list = []
            diff_list = []
            for j in range(i+1, lines.shape[0]):

                if mask[j] is True:
                    continue
                diff = np.square(line - lines[j,:4])
                diff_1 = np.sqrt(diff[0] + diff[1])
                diff_2 = np.sqrt(diff[2] + diff[3])
                diff = (diff_1 + diff_2) / 2.0

                if diff < 10:
                    idx_list.append(j)
                    diff_list.append(diff)

            if len(idx_list)==0:
                merged_lines.append(line)
            else:
                diff_list = np.array(diff_list)
                min_idx = idx_list[diff_list.argmin()]
                l = (line+lines[min_idx])/2
                merged_lines.append(l)
                mask[min_idx] = True
            mask[i] = True
        return np.array(merged_lines).astype(int)
        

    def draw_lines(self, img, img_name, lines, line_labels=None):
        if lines is not None:
            for i, line in enumerate(lines):
                x1, y1, x2, y2 = line.astype(int)
                if line_labels is not None:
                    if line_labels[i]:
                        cv2.line(img, (x1,y1), (x2,y2), (0,0,255), 1)   # bgr
                    else:
                        cv2.line(img, (x1,y1), (x2,y2), (0,255,0), 1)   # bgr
                else:
                    cv2.line(img, (x1,y1), (x2,y2), (0,255,0), 1)       # bgr
        # cv2.imwrite(f"D://mattab_kara//{img_name}", img)
        cv2.imshow(f'{img_name}', img)
        cv2.waitKey()


    def _filter_tilt(self, lines):
        new_lines = []
        for i, line in enumerate(lines):
            x1, x2, y1, y2 = line

            if abs(x1 - y1) <= 2:
                ave = int((x1 + y1) / 2)
                x1 = ave
                y1 = ave
                new_lines.append([x1, x2, y1, y2])
            elif abs(x2 - y2) <= 2:
                ave = int((x2+y2)/2)
                x2 = ave
                y2 = ave
                new_lines.append([x1, x2, y1, y2])
        
        return np.array(new_lines)

    
    def _format_line(self, lines):
        """
        将垂直线段小坐标转换成大坐标
        """
        for i, line in enumerate(lines):
            x1, x2, y1, y2 = line
            if x1 == y1:
                if x2 > y2:
                    line[1] = y2
                    line[3] = x2 
            if x2 == y2:
                if x1 > y1:
                    line[0] = y1
                    line[2] = x1           
        return lines
        
    def _generate_line_bboxes(self, lines: np.array, img_shape):
        fac = 4.0
        line_bboxes = np.empty((lines.shape), dtype=np.float32)
        for i, line in enumerate(lines):
            x1, y1, x2, y2 = line
            xc, yc = (x1+x2)/2.0, (y1+y2)/2.0
            if y1 == y2:    # 横线
                weig = x2 - x1 + 30.0
                high = weig / fac
            elif x1 == x2:
                high = y2 - y1 + 30.0
                weig = high / fac
            else:
                continue
            bx1 = xc - weig / 2.0 if xc - weig / 2.0 > 0 else 0 + 5
            by1 = yc - high / 2.0 if yc - high / 2.0 > 0 else 0 + 5
            bx2 = xc + weig / 2.0 if xc + weig / 2.0 < img_shape[1] else img_shape[1] - 5
            by2 = yc + high /2.0 if yc + high /2.0 < img_shape[0] else img_shape[0] - 5
            line_bboxes[i] = np.array([bx1, by1, bx2, by2])

        return line_bboxes

    def __call__(self, results):
        img = results['img']
        img_name = os.path.basename(results['ori_filename'])
        height = results['img_shape'][0]
        width = results['img_shape'][1]
        gt_bboxes = results.get('gt_bboxes')

        # hough
        houghimg = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        houghimg = cv2.GaussianBlur(houghimg,(3,3),0)
        edges = cv2.Canny(houghimg, 150, 350, apertureSize = 5)
        lines = cv2.HoughLinesP(edges,1,np.pi/180,threshold=100,minLineLength=int(width/5),maxLineGap=5)

        # kara
        # houghimg = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # houghimg = cv2.GaussianBlur(houghimg,(3,3),0)
        # edges = cv2.Canny(houghimg, 150, 350, apertureSize = 5)
        # lines = cv2.HoughLinesP(edges,1,np.pi/180,threshold=100,minLineLength=200,maxLineGap=5)

        # fld
        # gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # fld = cv2.ximgproc.createFastLineDetector()
        # lines = fld.detect(gray_img)


        # lsd = cv2.createLineSegmentDetector(0)
        # lines = lsd.detect(gray_img)
        # if lines is not None:
        #     lines = lines[0]

        # canny line
        # lines = None
        # for img_inf in self.canny_lines:
        #     if img_inf['file_name'] == img_name:
        #         lines = np.asarray(img_inf['line'])
        #         lines = lines[:,0:4].astype(int)
        #         lines = lines[:,np.newaxis,:]
        #         break

        if lines is not None:
            lines = np.squeeze(lines, axis=1)                       # nx1x4 -> nx4
            lines = self._filter_tilt(lines)
            lines = self._format_line(lines)
        else:
            lines = np.array([[width/4,height/2,width/2,height/2],],dtype=np.int32)
        
        lines = self.merge_line(lines)

        if lines.size == 0:
            lines = np.array([[width/4,height/2,width/2,height/2],],dtype=np.int32)


        # self.draw_lines(img, img_name, lines)


        # houghimg = np.zeros((height, width,3), np.uint8)
        # houghimg.fill(255)
        # houghimg = cv2.cvtColor(houghimg, cv2.COLOR_RGB2BGR)
        # self.draw_lines(img, img_name, lines, line_labels=line_labels)

        line_bboxes = self._generate_line_bboxes(lines, img_shape=results['img_shape'])
        
        # self.draw_line_bboxes(houghimg, img_name, line_bboxes)

        results['ori_lines'] = lines
        results['line_bboxes'] = line_bboxes
        return results


@PIPELINES.register_module()
class LoadSignedLines(LoadLines):
    def draw_line_bboxes(self, img, img_name, line_bboxes):
        if line_bboxes is not None:
            for i, line_box in enumerate(line_bboxes):
                x1, y1, x2, y2 = line_box.astype(int)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 1)    # bgr
        cv2.imshow(f'{img_name}', img)
        cv2.waitKey()

    def _b2l(self, bboxes:np.array):
        num_boxes = bboxes.shape[0]
        gt_lines = np.empty((num_boxes*4, 4), dtype=np.float32)
        for i, box in enumerate(bboxes):
            x1, y1, x2, y2 = bboxes[i,0], bboxes[i,1], bboxes[i,2], bboxes[i,3]

            gt_lines[i*4+0] = x1, y1, x2, y1
            gt_lines[i*4+1] = x2, y1, x2, y2
            gt_lines[i*4+2] = x1, y2, x2, y2
            gt_lines[i*4+3] = x1, y1, x1, y2
        return gt_lines

    def _classify_lines(self, lines:np.array, gt_bboxes:np.array):
        if gt_bboxes is None:
            line_labels = np.full(lines.shape[0], False, dtype=bool)
            return line_labels

        gt_lines = self._b2l(gt_bboxes)
        line_labels = np.full(lines.shape[0], False, dtype=bool)
        for i, line in enumerate(lines):
            l_x1, l_y1, l_x2, l_y2 = line
            for gt_line in gt_lines:
                gt_x1, gt_y1, gt_x2, gt_y2 = gt_line
                d = np.sum(np.square(np.array([l_x1, l_y1]) - np.array([gt_x1, gt_y1]))) \
                    + np.sum(np.square(np.array(l_x2, l_y2) - np.array(gt_x2, gt_y2)))
                if d <= 10.0:
                    line_labels[i] = True
        return line_labels


    def __call__(self, results):
        img = results['img']
        img_name = results['ori_filename']
        height = results['img_shape'][0]
        width = results['img_shape'][1]
        gt_bboxes = results.get('gt_bboxes')

        # hough 变换
        houghimg = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        houghimg = cv2.GaussianBlur(houghimg,(3,3),0)
        edges = cv2.Canny(houghimg, 150, 350, apertureSize = 5)
        lines = cv2.HoughLinesP(edges,1,np.pi/180,threshold=100,minLineLength=int(width/5),maxLineGap=5)
        
        # fld
        # gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # fld = cv2.ximgproc.createFastLineDetector()
        # lines = fld.detect(gray_img)

        # canny line
        # lines = None
        # for img_inf in self.canny_lines:
        #     if img_inf['file_name'] == img_name:
        #         lines = np.asarray(img_inf['line'])
        #         lines = lines[:,0:4].astype(int)
        #         lines = lines[:,np.newaxis,:]
        #         break

        if lines is not None:
            lines = np.squeeze(lines, axis=1)                       # nx1x4 -> nx4
            lines = self._filter_tilt(lines)
            lines = self._format_line(lines)
            # lines = torch.as_tensor(lines, dtype=torch.float32)
        else:
            lines = np.array([[width/4,height/2,width/2,height/2],],dtype=np.int32)
        
        
        lines = self.merge_line(lines)
        
        if lines.size == 0:
            lines = np.array([[width/4,height/2,width/2,height/2],],dtype=np.int32)

        
        # houghimg = np.zeros((height, width,3), np.uint8)
        # houghimg.fill(255)
        # houghimg = cv2.cvtColor(houghimg, cv2.COLOR_RGB2BGR)
        line_labels = self._classify_lines(lines, gt_bboxes)
        # self.draw_lines(img, img_name, lines, line_labels=line_labels)

        line_bboxes = self._generate_line_bboxes(lines, img_shape=results['img_shape'])
        # self.draw_line_bboxes(houghimg, img_name, line_bboxes)

        results['ori_lines'] = lines
        results['line_labels'] = line_labels
        results['line_bboxes'] = line_bboxes
        return results


@PIPELINES.register_module()
class MyResize(Resize):
    def _resize_line_bboxes(self, results):
        """Resize lines with ``results['scale_factor']``."""
        line_bboxes = results.get('line_bboxes')
        if line_bboxes is not None:
            line_bboxes = line_bboxes * results['scale_factor']

            if self.bbox_clip_border:
                img_shape = results['img_shape']
                line_bboxes[:, 0::2] = np.clip(line_bboxes[:, 0::2], 0, img_shape[1])
                line_bboxes[:, 1::2] = np.clip(line_bboxes[:, 1::2], 0, img_shape[0])

        results['line_bboxes'] = line_bboxes

    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map and line bboxes.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor', \
                'keep_ratio' keys are added into result dict.
        """

        if 'scale' not in results:
            if 'scale_factor' in results:
                img_shape = results['img'].shape[:2]
                scale_factor = results['scale_factor']
                assert isinstance(scale_factor, float)
                results['scale'] = tuple(
                    [int(x * scale_factor) for x in img_shape][::-1])
            else:
                self._random_scale(results)
        else:
            if not self.override:
                assert 'scale_factor' not in results, (
                    'scale and scale_factor cannot be both set.')
            else:
                results.pop('scale')
                if 'scale_factor' in results:
                    results.pop('scale_factor')
                self._random_scale(results)

        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_masks(results)
        self._resize_seg(results)
        self._resize_line_bboxes(results)
        return results


@PIPELINES.register_module()
class MyRandomFlip(RandomFlip):
    def __call__(self, results):
        if 'flip' not in results:
            if isinstance(self.direction, list):
                # None means non-flip
                direction_list = self.direction + [None]
            else:
                # None means non-flip
                direction_list = [self.direction, None]

            if isinstance(self.flip_ratio, list):
                non_flip_ratio = 1 - sum(self.flip_ratio)
                flip_ratio_list = self.flip_ratio + [non_flip_ratio]
            else:
                non_flip_ratio = 1 - self.flip_ratio
                # exclude non-flip
                single_ratio = self.flip_ratio / (len(direction_list) - 1)
                flip_ratio_list = [single_ratio] * (len(direction_list) -
                                                    1) + [non_flip_ratio]

            cur_dir = np.random.choice(direction_list, p=flip_ratio_list)

            results['flip'] = cur_dir is not None
        if 'flip_direction' not in results:
            results['flip_direction'] = cur_dir
        if results['flip']:
            # flip image
            for key in results.get('img_fields', ['img']):
                results[key] = mmcv.imflip(
                    results[key], direction=results['flip_direction'])
            # flip bboxes
            for key in results.get('bbox_fields', []):
                results[key] = self.bbox_flip(results[key],
                                              results['img_shape'],
                                              results['flip_direction'])
            # flip masks
            for key in results.get('mask_fields', []):
                results[key] = results[key].flip(results['flip_direction'])

            # flip segs
            for key in results.get('seg_fields', []):
                results[key] = mmcv.imflip(
                    results[key], direction=results['flip_direction'])
            # flip line_bboxes
            results['line_bboxes'] = self.bbox_flip(results['line_bboxes'],
                                          results['img_shape'],
                                          results['flip_direction'])
        return results


@PIPELINES.register_module()
class MyFormatBundle(DefaultFormatBundle):
    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with \
                default bundle.
        """

        if 'img' in results:
            img = results['img']
            if self.img_to_float is True and img.dtype == np.uint8:
                # Normally, image is of uint8 type without normalization.
                # At this time, it needs to be forced to be converted to
                # flot32, otherwise the model training and inference
                # will be wrong. Only used for YOLOX currently .
                img = img.astype(np.float32)
            # add default meta keys
            results = self._add_default_meta_keys(results)
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            results['img'] = DC(
                to_tensor(img), padding_value=self.pad_val['img'], stack=True)
        for key in ['proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels']:
            if key not in results:
                continue
            results[key] = DC(to_tensor(results[key]))
        if 'gt_masks' in results:
            results['gt_masks'] = DC(
                results['gt_masks'],
                padding_value=self.pad_val['masks'],
                cpu_only=True)
        if 'gt_semantic_seg' in results:
            results['gt_semantic_seg'] = DC(
                to_tensor(results['gt_semantic_seg'][None, ...]),
                padding_value=self.pad_val['seg'],
                stack=True)
        if 'line_bboxes' in results:
            results['line_bboxes'] = DC(to_tensor(results['line_bboxes']))
        if 'line_labels' in results:
            results['line_labels'] = DC(to_tensor(results['line_labels']))

        return results


@PIPELINES.register_module()
class LinesCollect(Collect):
    def __call__(self, results):
        """Call function to collect keys in results. The keys in ``meta_keys``
        will be converted to :obj:mmcv.DataContainer.

        Args:
            results (dict): Result dict contains the data to collect.

        Returns:
            dict: The result dict contains the following keys

                - keys in``self.keys``
                - ``img_metas``
        """

        data = {}
        img_meta = {}
        for key in self.meta_keys:
            img_meta[key] = results[key]
        img_meta['ori_lines'] = results.get('ori_lines')
        img_meta['line_labels'] = results.get('line_labels')
        img_meta['line_bboxes'] = results.get('line_bboxes')
        data['img_metas'] = DC(img_meta, cpu_only=True)
        for key in self.keys:
            data[key] = results[key]
        return data

