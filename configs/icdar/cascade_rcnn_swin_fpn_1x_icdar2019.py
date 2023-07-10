_base_ = '../cascade_rcnn/cascade_rcnn_swin_fpn_1x_coco.py'

custom_imports=dict(
    imports=['mmdet.models.roi_heads.refine_cascade_roi_head'])

# 我们需要对头中的类别数量进行修改来匹配数据集的标注


# 修改数据集相关设置
dataset_type = 'ICDAR_2019Dataset'
classes = ('table',)
data = dict(
    train=dict(
        type=dataset_type,
        img_prefix='data/ICDAR2019_cTDaR/training/TRACKA/ground_truth',
        classes=classes,
        ann_file='data/ICDAR2019_cTDaR/training/TRACKA/ground_truth/table.json'),
    val=dict(
        type=dataset_type,
        img_prefix='data/ICDAR2019_cTDaR/test/TRACKA',
        classes=classes,
        ann_file='data/ICDAR2019_cTDaR/test_ground_truth/table.json'),
    test=dict(
        type=dataset_type,
        img_prefix='data/ICDAR2019_cTDaR/test/TRACKA',
        classes=classes,
        ann_file='data/ICDAR2019_cTDaR/test_ground_truth/table.json')
)
evaluation = dict(metric=['bbox'])

# 我们可以使用预训练的 Mask R-CNN 来获取更好的性能
load_from = 'checkpoints/cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth'