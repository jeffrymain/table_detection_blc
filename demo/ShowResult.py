import imp
import torch
import os
from mmdet.apis import inference_detector, init_detector, show_result_pyplot

from pycocotools.coco import COCO

# 根据colab状态设置device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 选择模型对应的配置文件
# config = './configs/material/mask_rcnn_r50_fpn_1x_material_refine.py'
# config = './configs/material/faster_rcnn_r50_fpn_1x_material_refine.py'
config = './configs/material/cascade_mask_rcnn_r50_fpn_1x_material_refine.py'
# 选择下载好的checkpoint
# checkpoint = './work_dirs/mask_rcnn_r50_fpn_1x_material/epoch_12.pth'
checkpoint = './work_dirs/cascade_mask_rcnn_r50_fpn_1x_material_refine/epoch_12.pth'
# 初始化模型
model = init_detector(config, checkpoint, device=device)
img_root = './data/material_papers/img'
annFile = './data/material_papers/ann/test.json'
coco = COCO(annFile)
cats = coco.loadCats(coco.getCatIds())
catIds = coco.getCatIds(catNms=['table'])
imgIds = coco.getImgIds(catIds=catIds)


img_info = coco.loadImgs(imgIds[106])[0]
img_path = os.path.join(img_root, img_info['file_name'])
result = inference_detector(model, img_path)
result = result[0]
show_result_pyplot(model, img_path, result, score_thr=0.8, wait_time=2, title='idx')

img_info = coco.loadImgs(imgIds[111])[0]
img_path = os.path.join(img_root, img_info['file_name'])
result = inference_detector(model, img_path)
result = result[0]
show_result_pyplot(model, img_path, result, score_thr=0.8, wait_time=2, title='idx')

img_info = coco.loadImgs(imgIds[78])[0]
img_path = os.path.join(img_root, img_info['file_name'])
result = inference_detector(model, img_path)
result = result[0]
show_result_pyplot(model, img_path, result, score_thr=0.8, wait_time=2, title='idx')

img_info = coco.loadImgs(imgIds[80])[0]
img_path = os.path.join(img_root, img_info['file_name'])
result = inference_detector(model, img_path)
result = result[0]
show_result_pyplot(model, img_path, result, score_thr=0.8, wait_time=2, title='idx')

img_info = coco.loadImgs(imgIds[81])[0]
img_path = os.path.join(img_root, img_info['file_name'])
result = inference_detector(model, img_path)
result = result[0]
show_result_pyplot(model, img_path, result, score_thr=0.8, wait_time=2, title='idx')

for idx in imgIds:
    img_info = coco.loadImgs(idx)[0]
    img_path = os.path.join(img_root, img_info['file_name'])

    result = inference_detector(model, img_path)
    result = result[0]
    show_result_pyplot(model, img_path, result, score_thr=0.8, wait_time=2, title=idx)