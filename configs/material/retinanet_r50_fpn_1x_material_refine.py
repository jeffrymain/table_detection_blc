_base_ = '../retinanet/retinanet_r50_fpn_1x_coco.py'

custom_imports=dict(
    imports=['mmdet.models.dense_heads.refine_retina_head'])

# 我们需要对头中的类别数量进行修改来匹配数据集的标注
model = dict(
    bbox_head=dict(
        type='RefineRetinaHead',
        num_classes=1,                                      # 修改预测类别
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)
    ),

    test_cfg = dict(
        use_refine = False,
    )
)

# 修改数据集相关设置
dataset_type = 'CocoDataset'
classes = ('table',)
data = dict(
    train=dict(
        img_prefix='data/material_papers/img',
        classes=classes,
        ann_file='data/material_papers/ann/train.json'),
    val=dict(
        img_prefix='data/material_papers/img',
        classes=classes,
        ann_file='data/material_papers/ann/val.json'),
    test=dict(
        img_prefix='data/material_papers/img',
        classes=classes,
        ann_file='data/material_papers/ann/test.json')
)

# 我们可以使用预训练的 Faster R-CNN 来获取更好的性能
load_from = 'checkpoints/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth'


