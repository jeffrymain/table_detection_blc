_base_ = '../cascade_rcnn/cascade_mask_rcnn_hrnetv2p_w32_20e_coco.py'


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

# 我们可以使用预训练的 Mask R-CNN 来获取更好的性能
# load_from = 'checkpoints/cascade_mask_rcnn_r50_fpn_1x_coco_20200203-9d4dcb24.pth'

