from asyncio.windows_events import NULL
import os
from re import M
from unittest import result
from matplotlib import image
import numpy as np
import torch
from PIL import Image
import cv2
from .coco import CocoDataset
import tempfile
import os.path as osp
from .builder import DATASETS
import mmcv

from mmcv.utils import print_log

class HoughLineMixin:
    def _format_vline(self, lines):
        """
        将垂直线段小坐标转换成大坐标
        """
        for i, line in enumerate(lines):
            x1, x2, y1, y2 = line
            if x1 == y1:
                line[1] = y2
                line[3] = x2            
        return lines

    def _load_bboxes(self, results):
        """Private function to load bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded bounding box annotations.
        """

        ann_info = results['ann_info']
        results['gt_bboxes'] = ann_info['bboxes'].copy()

        if self.denorm_bbox:
            h, w = results['img_shape'][:2]
            bbox_num = results['gt_bboxes'].shape[0]
            if bbox_num != 0:
                results['gt_bboxes'][:, 0::2] *= w
                results['gt_bboxes'][:, 1::2] *= h
            results['gt_bboxes'] = results['gt_bboxes'].astype(np.float32)

        gt_bboxes_ignore = ann_info.get('bboxes_ignore', None)
        if gt_bboxes_ignore is not None:
            results['gt_bboxes_ignore'] = gt_bboxes_ignore.copy()
            results['bbox_fields'].append('gt_bboxes_ignore')
        results['bbox_fields'].append('gt_bboxes')

        gt_is_group_ofs = ann_info.get('gt_is_group_ofs', None)
        if gt_is_group_ofs is not None:
            results['gt_is_group_ofs'] = gt_is_group_ofs.copy()

        return results

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
                if d <= 30.0:
                    line_labels[i] = True
                    break
        return line_labels

def xywh2xyxy(ann):
    x1 = ann[0]
    y1 = ann[1]
    x2 = ann[0] + ann[2] - 1
    y2 = ann[1] + ann[3] - 1
    return [x1, y1, x2, y2]


@DATASETS.register_module()
class ICDAR_2019Dataset(CocoDataset, HoughLineMixin):
    CLASSES = ('table')

    def _line_cls2json(self, results):
        lcls_json_result = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            line_result = results[idx]

            for label in range(len(line_result)):
                for i in range(line_result[label][0].shape[0]):
                    data = dict()
                    data['img_id'] = img_id
                    data['ori_line'] = line_result[label][0][i].tolist()
                    data['line_box'] = line_result[label][1][i][:4].tolist()
                    data['score'] = line_result[label][1][i][4]
                    data['label'] = label
                    lcls_json_result.append(data)

        return lcls_json_result

    def line_results2json(self, results, jsonfile_prefix=None):

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        # 线段结果存进文件
        result_files = dict()
        json_results = self._line_cls2json(results)

        result_files['line_result'] = f'{jsonfile_prefix}.line_ressult.json'

        mmcv.dump(json_results, result_files['line_result'])
        return result_files, tmp_dir


    def evaluate_line_cls(self, result_files, coco_gt):
        try:
            predictions = mmcv.load(result_files['line_result'])
        except IndexError:
            # TODO: 接入logger
            print('The testing results of the whole dataset is empty.')

        tp = 0
        fp = 0
        fn = 0
        tn = 0
        cls_results = dict()
        for i, data in enumerate(predictions):
            img_id = data['img_id']
            ori_line = np.array(data['ori_line']).reshape(1,4)
            annIds = coco_gt.getAnnIds(img_id)
            anns = coco_gt.loadAnns(annIds)
            gt_boxes = np.array([xywh2xyxy(ann['bbox']) for ann in anns])
            gt_label = self._classify_lines(ori_line, gt_boxes)
            label = bool(data['label'])
            if label:
                if gt_label:
                    tp += 1
                else:
                    fp += 1
            else:
                if gt_label:
                    fn += 1
                else:
                    tn += 1

        cls_results['tp'] = tp
        cls_results['fp'] = fp
        cls_results['fn'] = fn
        cls_results['tn'] = tn
        try:
            cls_results['Accuracy'] = float(tp + tn) / (tp + fp + fn + tn)
        except:
            cls_results['Accuracy'] = float(-1)
        try:
            cls_results['Precision'] = float(tp) / (tp + fp)
        except:
            cls_results['Precision'] = float(-1)
        try:
            cls_results['Recall'] = float(tp) / (tp + fn)
        except:
            cls_results['Recall'] = float(-1)
        try:
            cls_results['F-Score'] = (1 + 1) * cls_results['Precision'] * cls_results['Recall'] / (1 * (cls_results['Precision'] + cls_results['Recall']))
        except:
            cls_results['F-Score'] = float(-1)
        return cls_results

    def evaluate_lines(
        self,
        runner,
        results,
        jsonfile_prefix=None,
        classwise=False
    ):
        """ 评价线段分类
        """
        coco_gt = self.coco

        result_files, tmp_dir = self.line_results2json(results, jsonfile_prefix)

        eval_results = self.evaluate_line_cls(result_files, coco_gt)

        msg = f'Evaluating Lines...'
        print_log(msg, logger=runner.logger)

        msg = f"\nTP = {eval_results['tp']} | FP = {eval_results['fp']}\n"
        msg += f"FN = {eval_results['fn']} | TN = {eval_results['tn']}\n"
        msg += f"Accuracy = {eval_results['Accuracy']} | Precision = {eval_results['Precision']}\n"
        msg += f"Recall   = {eval_results['Recall']} | F-Score   = {eval_results['F-Score']}\n"
        print_log(msg, logger=runner.logger)

        for name, val in eval_results.items():
            runner.log_buffer.output[name] = val
        
        
        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results
