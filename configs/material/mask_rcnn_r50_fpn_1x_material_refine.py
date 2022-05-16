_base_ = '../mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py'

custom_imports=dict(
    imports=['mmdet.models.roi_heads.refine_roi_head'])

# 我们需要对头中的类别数量进行修改来匹配数据集的标注
model = dict(
    roi_head=dict(  # RoIHead 封装了两步(two-stage)/级联(cascade)检测器的第二步。
        type='RefineRoIHead',  # RoI head 的类型，这是我自己定义的ROI head。更多细节请参考 https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/roi_heads/standard_roi_head.py#L10。
        bbox_roi_extractor=dict(  # 用于 bbox 回归的 RoI 特征提取器。
            type='SingleRoIExtractor',  # RoI 特征提取器的类型，大多数方法使用  SingleRoIExtractor，更多细节请参考 https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/roi_heads/roi_extractors/single_level.py#L10。
            roi_layer=dict(  # RoI 层的配置
                type='RoIAlign',  # RoI 层的类别, 也支持 DeformRoIPoolingPack 和 ModulatedDeformRoIPoolingPack，更多细节请参考 https://github.com/open-mmlab/mmdetection/blob/master/mmdet/ops/roi_align/roi_align.py#L79。
                output_size=7,  # 特征图的输出大小。
                sampling_ratio=0),  # 提取 RoI 特征时的采样率。0 表示自适应比率。
            out_channels=256,  # 提取特征的输出通道。
            featmap_strides=[4, 8, 16, 32]),  # 多尺度特征图的步幅，应该与主干的架构保持一致。
        bbox_head=dict(  # RoIHead 中 box head 的配置.
            type='Shared2FCBBoxHead',  # bbox head 的类别，更多细节请参考 https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/roi_heads/bbox_heads/convfc_bbox_head.py#L177。
            in_channels=256,  # bbox head 的输入通道。 这与 roi_extractor 中的 out_channels 一致。
            fc_out_channels=1024,  # FC 层的输出特征通道。
            roi_feat_size=7,  # 候选区域(Region of Interest)特征的大小。
            num_classes=1,  # 分类的类别数量。
            bbox_coder=dict(  # 第二阶段使用的框编码器。
                type='DeltaXYWHBBoxCoder',  # 框编码器的类别，大多数情况使用 'DeltaXYWHBBoxCoder'。
                target_means=[0.0, 0.0, 0.0, 0.0],  # 用于编码和解码框的均值
                target_stds=[0.1, 0.1, 0.2, 0.2]),  # 编码和解码的标准差。因为框更准确，所以值更小，常规设置时 [0.1, 0.1, 0.2, 0.2]。
            reg_class_agnostic=False,  # 回归是否与类别无关。
            loss_cls=dict(  # 分类分支的损失函数配置
                type='CrossEntropyLoss',  # 分类分支的损失类型，我们也支持 FocalLoss 等。
                use_sigmoid=False,  # 是否使用 sigmoid。
                loss_weight=1.0),  # 分类分支的损失权重。
            loss_bbox=dict(  # 回归分支的损失函数配置。
                type='L1Loss',  # 损失类型，我们还支持许多 IoU Losses 和 Smooth L1-loss 等。
                loss_weight=1.0)),  # 回归分支的损失权重。
        mask_roi_extractor=dict(  # 用于 mask 生成的 RoI 特征提取器。
            type='SingleRoIExtractor',  # RoI 特征提取器的类型，大多数方法使用 SingleRoIExtractor。
            roi_layer=dict(  # 提取实例分割特征的 RoI 层配置
                type='RoIAlign',  # RoI 层的类型，也支持 DeformRoIPoolingPack 和 ModulatedDeformRoIPoolingPack。
                output_size=14,  # 特征图的输出大小。
                sampling_ratio=0),  # 提取 RoI 特征时的采样率。
            out_channels=256,  # 提取特征的输出通道。
            featmap_strides=[4, 8, 16, 32]),  # 多尺度特征图的步幅。
        mask_head=dict(  # mask 预测 head 模型
            type='FCNMaskHead',  # mask head 的类型，更多细节请参考 https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/roi_heads/mask_heads/fcn_mask_head.py#L21。
            num_convs=4,  # mask head 中的卷积层数
            in_channels=256,  # 输入通道，应与 mask roi extractor 的输出通道一致。
            conv_out_channels=256,  # 卷积层的输出通道。
            num_classes=1,  # 要分割的类别数。
            loss_mask=dict(  # mask 分支的损失函数配置。
                type='CrossEntropyLoss',  # 用于分割的损失类型。
                use_mask=True,  # 是否只在正确的类中训练 mask。
                loss_weight=1.0))),  # mask 分支的损失权重.
    test_cfg = dict(
        rcnn = dict(
            use_refine = True,
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

# 我们可以使用预训练的 Mask R-CNN 来获取更好的性能
load_from = 'checkpoints/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth'
