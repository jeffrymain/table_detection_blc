_base_ = '../cascade_rcnn/cascade_rcnn_hrnetv2p_w32_20e_coco.py'


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

