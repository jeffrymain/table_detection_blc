_base_ = '../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'

custom_imports=dict(
    imports=['mmdet.models.roi_heads.refine_roi_head'])

# 我们需要对头中的类别数量进行修改来匹配数据集的标注
model = dict(

    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(num_classes=1))

)
# model settings
model = dict(
    roi_head=dict(
        type='RefineRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]
        ),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)
        ),
    ),
    test_cfg = dict(
        rcnn = dict(
            use_refine = False,
        )
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
load_from = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'


