_base_ = '../cascade_rcnn/cascade_mask_rcnn_r50_fpn_2x_coco.py'

custom_imports=dict(
    imports=['mmdet.models.roi_heads.refine_cascade_roi_head'])

# 我们需要对头中的类别数量进行修改来匹配数据集的标注
model = dict(
    roi_head=dict(
        type='RefineCascadeRoIHead',
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ],
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=1,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)),
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
        )
    ),

    test_cfg = dict(
        rcnn = dict(
            use_refine = True,
            cls_line_cfg = dict(
                pos_thr = 0.8,  # 目前还没用到这个阈值
            )
        )
    )
)

# 修改数据集相关设置
dataset_type = 'ICDAR_2017Dataset'
classes = ('table',)
data = dict(
    train=dict(
        type=dataset_type,
        img_prefix='data/ICDAR2017_POD/Images/TrainSet',
        classes=classes,
        ann_file='data/ICDAR2017_POD/Annotations/TrainSet/train.json'),
    val=dict(
        type=dataset_type,
        img_prefix='data/ICDAR2017_POD/Images/TestSet',
        classes=classes,
        ann_file='data/ICDAR2017_POD/Annotations/TestSet/test.json'),
    test=dict(
        type=dataset_type,
        img_prefix='data/ICDAR2017_POD/Images/TestSet',
        classes=classes,
        ann_file='data/ICDAR2017_POD/Annotations/TestSet/test.json')
)
evaluation = dict(metric=['bbox'])

# 我们可以使用预训练的 Mask R-CNN 来获取更好的性能
load_from = 'checkpoints/cascade_mask_rcnn_r50_fpn_1x_coco_20200203-9d4dcb24.pth'

