# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from base64 import encode
from cgitb import reset
import os
import os.path as osp
import time
import warnings
import sys
sys.path.append(os.getcwd())

import numpy as np
import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet.apis import multi_gpu_test, single_gpu_test, my_single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.utils import setup_multi_processes, update_data_root


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)

    # update data root according to MMDET_DATASETS
    update_data_root(cfg)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if 'pretrained' in cfg.model:
        cfg.model.pretrained = None
    elif 'init_cfg' in cfg.model.backbone:
        cfg.model.backbone.init_cfg = None

    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed testing. Use the first GPU '
                      'in `gpu_ids` now.')
    else:
        cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    rank, _ = get_dist_info()
    # allows not to create
    if args.work_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        json_file = osp.join(args.work_dir, f'eval_{timestamp}.json')

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    if not distributed:
        model = MMDataParallel(model, device_ids=cfg.gpu_ids)
        # 原生
        # outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
        #                           args.show_score_thr)

        # line result
        outputs, line_results = my_single_gpu_test(model, data_loader, args.show, args.show_dir,
                                  args.show_score_thr)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule', 'dynamic_intervals'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))

            # line detection 对output做的处理
            for i,img in enumerate(outputs):
                bboxes = img[0]
                # 不同种类bboxes的列表
                if isinstance(bboxes, list):
                    for j,cat_bboxes in enumerate(bboxes):
                        inds = np.where(cat_bboxes[:,4]>=0.0)
                        outputs[i][0][j] = cat_bboxes[inds]
                else:
                    inds = np.where(bboxes[:,4]>=0.0)
                    outputs[i][0] = bboxes[inds]

            # 原生代码
            metric = dataset.evaluate(outputs, **eval_kwargs)

            # _ = dataset.evaluate_lines()

            # 修正结果
            refine_outputs = refine_bboxes(outputs, line_results)
            metric_refine = dataset.evaluate(refine_outputs, **eval_kwargs)
            print(metric)
            # 打印修正后结果
            print("\n refine result\n")
            print(metric_refine)
            metric_dict = dict(config=args.config, metric=metric)
            if args.work_dir is not None and rank == 0:
                mmcv.dump(metric_dict, json_file)

# results 为模型输出
def refine_bboxes(results, line_results):
    for i, data in enumerate(results):
        # bboxes是一个列表，每个元素代表每种目标的预测框 ps:在有mask的时候
        bboxes = data[0]
        det_lines = line_results[i][1][1]
        det_lines[:,:4] = line_results[i][1][0]
        if isinstance(bboxes,list):# 有mask的情况
            table_bboxes = bboxes[0]
        else:
            table_bboxes = bboxes
        refine_bboxes = _refine_bboxes(table_bboxes, det_lines)
        if isinstance(bboxes,list):
            results[i][0][0] = refine_bboxes
        else:
            results[i][0] = refine_bboxes

    return results

# bbox为array(n,5) det_lines为(n,5)
def _refine_bboxes(bboxes:np, det_lines:np.array):
    # 每个边界框的中心点坐标
    if det_lines.shape[0] == 0:
        return bboxes
    cent = np.empty((bboxes.shape[0], 2))
    cent[:,0] = (bboxes[:,0] + bboxes[:,2]) / 2.0
    cent[:,1] = (bboxes[:,1] + bboxes[:,3]) / 2.0

    encode_bboxes = np.empty((bboxes.shape[0],4))
    encode_bboxes[:,0] = bboxes[:,1] - cent[:,1]
    encode_bboxes[:,1] = bboxes[:,2] - cent[:,0]
    encode_bboxes[:,2] = bboxes[:,3] - cent[:,1]
    encode_bboxes[:,3] = bboxes[:,0] - cent[:,0]

    match_mat = np.zeros((bboxes.shape[0]*4, 5))
    match_mat[:,:4] = _b2l(bboxes)
    for i, line in enumerate(match_mat):
        diff = np.square(det_lines[:,:4] - np.repeat(line[:4].reshape(1,4), det_lines.shape[0], axis=0))
        diff_1 = np.sqrt(diff[:,0] + diff[:,1])
        diff_2 = np.sqrt(diff[:,2] + diff[:,3])
        diff = (diff_1 + diff_2) / 2.0

        min_idx = diff.argmin()

        # TODO: 这里有个超参数
        if diff[min_idx] <= 15.0:
            match_mat[i] = det_lines[min_idx]

    match_mat[0::4, [0,1,3]] = match_mat[0::4, [1,2,0]] - cent[:,[1,0,0]]
    match_mat[0::4, 2] = encode_bboxes[:,2]
    match_mat[1::4, [0,1,2]] = match_mat[1::4, [1,0,3]] - cent[:,[1,0,1]]
    match_mat[1::4, 3] = encode_bboxes[:,3]
    match_mat[2::4, [1,2,3]] = match_mat[2::4, [2,1,0]] - cent[:,[0,1,0]]
    match_mat[2::4, 0] = encode_bboxes[:,0]
    match_mat[3::4, [0,2,3]] = match_mat[3::4, [1,3,0]] - cent[:,[1,1,0]]
    match_mat[3::4, 1] = encode_bboxes[:,1]

    # 先不使用概率
    for i, encode_bbox in enumerate(encode_bboxes):
        up = (match_mat[[0+i*4,1+i*4,3+i*4], 0] - np.repeat(encode_bbox[0], 3, axis=0))
        up = np.sum(up)
        div = np.sum(np.ceil(match_mat[[0+i*4,1+i*4,3+i*4], 4]))
        if div == 0.0:
            div = 1.0
        up = up / div + encode_bbox[0]

        right = (match_mat[[0+i*4,1+i*4,2+i*4], 1] - np.repeat(encode_bbox[1], 3, axis=0))
        right = np.sum(right)
        div = np.sum(np.ceil(match_mat[[0+i*4,1+i*4,2+i*4], 4]))
        if div == 0.0:
            div = 1.0
        right = right / div + encode_bbox[1]

        bottom = (match_mat[[1+i*4,2+i*4,3+i*4], 2] - np.repeat(encode_bbox[2], 3, axis=0))
        bottom = np.sum(bottom)
        div = np.sum(np.ceil(match_mat[[1+i*4,2+i*4,3+i*4], 4]))
        if div == 0.0:
            div = 1.0
        bottom = bottom / div + encode_bbox[2]

        left = (match_mat[[0+i*4,2+i*4,3+i*4], 3] - np.repeat(encode_bbox[3], 3, axis=0))
        left = np.sum(left)
        div = np.sum(np.ceil(match_mat[[0+i*4,2+i*4,3+i*4], 4]))
        if div == 0.0:
            div = 1.0
        left =  left / div + encode_bbox[3]

        encode_bboxes[i] = np.array([up, right, bottom, left])
        
    refine_bboxes = np.empty((bboxes.shape[0], 5))
    refine_bboxes[:, 0] = cent[:,0] + encode_bboxes[:,3]  # x1
    refine_bboxes[:, 1] = cent[:,1] + encode_bboxes[:,0]  # y1
    refine_bboxes[:, 2] = cent[:,0] + encode_bboxes[:,1]  # x2
    refine_bboxes[:, 3] = cent[:,1] + encode_bboxes[:,2]  # y2
    refine_bboxes[:, 4] = bboxes[:, 4]
    return refine_bboxes

# bboxes为(n,4)or(n,5)
# 输出为(n*4,4)
def _b2l(bboxes:np.array):
    num_boxes = bboxes.shape[0]
    gt_lines = np.empty((num_boxes*4, 4), dtype=np.float32)
    for i, box in enumerate(bboxes):
        x1, y1, x2, y2 = bboxes[i,0], bboxes[i,1], bboxes[i,2], bboxes[i,3]

        gt_lines[i*4+0] = x1, y1, x2, y1
        gt_lines[i*4+1] = x2, y1, x2, y2
        gt_lines[i*4+2] = x1, y2, x2, y2
        gt_lines[i*4+3] = x1, y1, x1, y2
    return gt_lines

if __name__ == '__main__':
    main()
