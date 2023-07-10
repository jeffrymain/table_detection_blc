_base_ = '../retinanet/retinanet_swin_fpn_1x_coco.py'

custom_imports=dict(
    imports=['mmdet.models.dense_heads.refine_retina_head'])

# 我们需要对头中的类别数量进行修改来匹配数据集的标注
model = dict(
    roi_head = dict(
        type='RefineRetinaHead',

        bbox_head=dict(
            type='RetinaHead',
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
        
        lbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(
                type='RoIAlign',
                output_size=7,
                sampling_ratio=0
            ),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]
        ),
        lbox_head=dict(
            type='ClassifyHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1,
            loss_cls=dict(
                type='CrossEntropyLoss',
                use_sigmoid=False, 
                loss_weight=1.0
            ),
        ),
    ),
    test_cfg = dict(
        use_refine = False,
    )
)

# 修改数据集相关设置
dataset_type = 'MaterialDataset'
classes = ('table',)
data = dict(
    train=dict(
        type=dataset_type,
        img_prefix='data/material_papers_e/img',
        classes=classes,
        ann_file='data/material_papers_e/ann/coco_formate_ann/train.json'),
    val=dict(
        type=dataset_type,
        img_prefix='data/material_papers_e/img',
        classes=classes,
        ann_file='data/material_papers_e/ann/coco_formate_ann/val.json'),
    test=dict(
        type=dataset_type,
        img_prefix='data/material_papers_e/img',
        classes=classes,
        ann_file='data/material_papers_e/ann/coco_formate_ann/test.json')
)

# 我们可以使用预训练的 Faster R-CNN 来获取更好的性能
load_from = 'checkpoints/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth'


