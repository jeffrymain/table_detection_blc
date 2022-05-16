# dataset settings
dataset_type = 'CocoDataset'            # 数据集类型，这将被用来定义数据集。
data_root = 'data/coco/'                # 数据的根路径。
img_norm_cfg = dict(                    # 图像归一化配置，用来归一化输入的图像。
    mean=[123.675, 116.28, 103.53],         # 预训练里用于预训练主干网络模型的平均值。
    std=[58.395, 57.12, 57.375],            # 预训练里用于预训练主干网络模型的标准差。
    to_rgb=True                             #  预训练里用于预训练主干网络的图像的通道顺序。
)
train_pipeline = [  # 训练流程
    dict(type='LoadImageFromFile'),     # 第 1 个流程，从文件路径里加载图像。
    dict(                               
        type='LoadAnnotations',         # 第 2 个流程，对于当前图像，加载它的注释信息。
        with_bbox=True,                     # 是否使用标注框(bounding box)， 目标检测需要设置为 True。
        with_mask=True,                     # 是否使用 instance mask，实例分割需要设置为 True。
        with_seg=True
        # poly2mask=False                   # 是否将 polygon mask 转化为 instance mask, 设置为 False 以加速和节省内存。
    ),
    dict(
        type='Resize',                  # 变化图像和其注释大小的数据增广的流程。
        img_scale=(1333, 800),              # 图像的最大规模。
        keep_ratio=True                     # 是否保持图像的长宽比。
    ),
    dict(
        type='RandomFlip',              # 翻转图像和其注释大小的数据增广的流程。
        flip_ratio=0.5                      # 翻转图像的概率。
    ),
    dict(
        type='Normalize',               # 归一化当前图像的数据增广的流程。
        **img_norm_cfg                      # 这些键与 img_norm_cfg 一致，因为 img_norm_cfg 被用作参数。
    ),
    dict(
        type='Pad',                     # 填充当前图像到指定大小的数据增广的流程。
        size_divisor=32                     # 填充图像可以被当前值整除。
    ),
    dict(
        type='SegRescale', 
        scale_factor=1 / 8
    ),
    dict(type='DefaultFormatBundle'),   # 流程里收集数据的默认格式捆。
    dict(
        type='Collect',                 # 决定数据中哪些键应该传递给检测器的流程
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),     # 第 1 个流程，从文件路径里加载图像。
    dict(
        type='MultiScaleFlipAug',       # 封装测试时数据增广(test time augmentations)。
        img_scale=(1333, 800),              # 决定测试时可改变图像的最大规模。用于改变图像大小的流程。
        flip=False,                         # 测试时是否翻转图像。
        transforms=[
            dict(
                type='Resize',           # 使用改变图像大小的数据增广。
                keep_ratio=True             # 是否保持宽和高的比例，这里的图像比例设置将覆盖上面的图像规模大小的设置。
            ),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,                      # 单个 GPU 的 Batch size
    workers_per_gpu=2,                      # 单个 GPU 分配的数据加载线程数
    train=dict(                             # 训练数据集配置
        type=dataset_type,                                                  # 数据集的类别, 更多细节请参考 https://github.com/open-mmlab/mmdetection/blob/master/mmdet/datasets/coco.py#L19。
        ann_file=data_root + 'annotations/instances_train2017.json',        # 注释文件路径
        img_prefix=data_root + 'train2017/',                                # 图片路径前缀
        seg_prefix=data_root + 'stuffthingmaps/train2017/',
        pipeline=train_pipeline),                                           # 流程, 这是由之前创建的 train_pipeline 传递的。
    val=dict(                               # 验证数据集的配置
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),                                            # 由之前创建的 test_pipeline 传递的流程。
    test=dict(                              # 测试数据集配置，修改测试开发/测试(test-dev/test)提交的 ann_file
        type=dataset_type,  
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))                                            # 由之前创建的 test_pipeline 传递的流程。
evaluation = dict(                          # evaluation hook 的配置，更多细节请参考 https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/evaluation/eval_hooks.py#L7。
    metric=['bbox', 'segm']                                                 # 验证期间使用的指标。
)
