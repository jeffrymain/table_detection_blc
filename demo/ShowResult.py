import imp
import os
import sys
import copy
import turtle
sys.path.append(os.getcwd())
import torch
import numpy as np
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
from pycocotools.coco import COCO
import cv2
import json

# both parameter are (5,) nparray
def in_area(bbox, inner, outter):
    if bbox[0]>inner[0] or bbox[1]>inner[1] or bbox[2]<inner[2] or bbox[3]<inner[3]:
        return False
    if bbox[0]<outter[0] or bbox[1]<outter[1] or bbox[2]>outter[2] or bbox[3]>outter[3]:
        return False
    return True

# table_box.shape = (4,) nparray
# gt_bboxes.shape = (n,4) nparray
# return.shape = (n,) list
def compu_iou(table_bbox, gt_bboxes):
    ious = []
    for gt_bbox in gt_bboxes:
        union = (min(table_bbox[0],gt_bbox[0]),min(table_bbox[1],gt_bbox[1]),max(table_bbox[2],gt_bbox[2]),max(table_bbox[3],gt_bbox[3]))
        inter = (max(table_bbox[0],gt_bbox[0]),max(table_bbox[1],gt_bbox[1]),min(table_bbox[2],gt_bbox[2]),min(table_bbox[3],gt_bbox[3]))
        if(inter[0]>=inter[2] or inter[1]>=inter[3]):
            inter = (0,0,0,0)
            
        ious.append(((inter[2]-inter[0])*(inter[3]-inter[1]))/((union[2]-union[0])*(union[3]-union[1])))
    return max(ious)
# bboxes为一张图片的bboxes，line_results为一张图片的line_results
# bboxes是一个列表，每个元素代表每种目标的预测框
def refine_bboxes(bboxes, line_results):
    det_lines = line_results[1][1]
    det_lines[:,:4] = line_results[1][0]
    # 这里取第一种
    table_bboxes = bboxes[0]
    refine_bboxes = _refine_bboxes(table_bboxes, det_lines)
    bboxes[0] = refine_bboxes

    return bboxes

# bbox为array(n,5) det_lines为(n,5)
def _refine_bboxes(bboxes:np, det_lines:np.array):
    # 每个边界框的中心点坐标
    if det_lines.shape[0] == 0:
        return bboxes
    cent = np.empty((bboxes.shape[0], 2))
    cent[:,0] = (bboxes[:,0] + bboxes[:,2]) / 2.0
    cent[:,1] = (bboxes[:,1] + bboxes[:,3]) / 2.0

    encode_bboxes = np.empty((bboxes.shape[0],4))
    encode_bboxes[:,0] = bboxes[:,1] - cent[:,1]
    encode_bboxes[:,1] = bboxes[:,2] - cent[:,0]
    encode_bboxes[:,2] = bboxes[:,3] - cent[:,1]
    encode_bboxes[:,3] = bboxes[:,0] - cent[:,0]

    match_mat = np.zeros((bboxes.shape[0]*4, 5))
    match_mat[:,:4] = _b2l(bboxes)
    for i, line in enumerate(match_mat):
        diff = np.square(det_lines[:,:4] - np.repeat(line[:4].reshape(1,4), det_lines.shape[0], axis=0))
        diff_1 = np.sqrt(diff[:,0] + diff[:,1])
        diff_2 = np.sqrt(diff[:,2] + diff[:,3])
        diff = (diff_1 + diff_2) / 2.0

        min_idx = diff.argmin()

        # TODO: 这里有个超参数
        if diff[min_idx] <= 10.0:
            match_mat[i] = det_lines[min_idx]

    match_mat[0::4, [0,1,3]] = match_mat[0::4, [1,2,0]] - cent[:,[1,0,0]]
    match_mat[0::4, 2] = encode_bboxes[:,2]
    match_mat[1::4, [0,1,2]] = match_mat[1::4, [1,0,3]] - cent[:,[1,0,1]]
    match_mat[1::4, 3] = encode_bboxes[:,3]
    match_mat[2::4, [1,2,3]] = match_mat[2::4, [2,1,0]] - cent[:,[0,1,0]]
    match_mat[2::4, 0] = encode_bboxes[:,0]
    match_mat[3::4, [0,2,3]] = match_mat[3::4, [1,3,0]] - cent[:,[1,1,0]]
    match_mat[3::4, 1] = encode_bboxes[:,1]

    # 先不使用概率
    for i, encode_bbox in enumerate(encode_bboxes):
        up = (match_mat[[0+i*4,1+i*4,3+i*4], 0] - np.repeat(encode_bbox[0], 3, axis=0))
        up = np.sum(up)
        div = np.sum(np.ceil(match_mat[[0+i*4,1+i*4,3+i*4], 4]))
        if div == 0.0:
            div = 1.0
        up = up / div + encode_bbox[0]

        right = (match_mat[[0+i*4,1+i*4,2+i*4], 1] - np.repeat(encode_bbox[1], 3, axis=0))
        right = np.sum(right)
        div = np.sum(np.ceil(match_mat[[0+i*4,1+i*4,2+i*4], 4]))
        if div == 0.0:
            div = 1.0
        right = right / div + encode_bbox[1]

        bottom = (match_mat[[1+i*4,2+i*4,3+i*4], 2] - np.repeat(encode_bbox[2], 3, axis=0))
        bottom = np.sum(bottom)
        div = np.sum(np.ceil(match_mat[[1+i*4,2+i*4,3+i*4], 4]))
        if div == 0.0:
            div = 1.0
        bottom = bottom / div + encode_bbox[2]

        left = (match_mat[[0+i*4,2+i*4,3+i*4], 3] - np.repeat(encode_bbox[3], 3, axis=0))
        left = np.sum(left)
        div = np.sum(np.ceil(match_mat[[0+i*4,2+i*4,3+i*4], 4]))
        if div == 0.0:
            div = 1.0
        left =  left / div + encode_bbox[3]

        encode_bboxes[i] = np.array([up, right, bottom, left])
        
    refine_bboxes = np.empty((bboxes.shape[0], 5))
    refine_bboxes[:, 0] = cent[:,0] + encode_bboxes[:,3]  # x1
    refine_bboxes[:, 1] = cent[:,1] + encode_bboxes[:,0]  # y1
    refine_bboxes[:, 2] = cent[:,0] + encode_bboxes[:,1]  # x2
    refine_bboxes[:, 3] = cent[:,1] + encode_bboxes[:,2]  # y2
    refine_bboxes[:, 4] = bboxes[:, 4]
    return refine_bboxes

# bboxes为(n,4)or(n,5)
# 输出为(n*4,4)
def _b2l(bboxes:np.array):
    num_boxes = bboxes.shape[0]
    gt_lines = np.empty((num_boxes*4, 4), dtype=np.float32)
    for i, box in enumerate(bboxes):
        x1, y1, x2, y2 = bboxes[i,0], bboxes[i,1], bboxes[i,2], bboxes[i,3]

        gt_lines[i*4+0] = x1, y1, x2, y1
        gt_lines[i*4+1] = x2, y1, x2, y2
        gt_lines[i*4+2] = x1, y2, x2, y2
        gt_lines[i*4+3] = x1, y1, x1, y2
    return gt_lines

# img 图片路径， line_result 该图片的分类结果
def draw_line_cls(img, line_result, out_file, thickness=1):
    img = cv2.imread(img)
    false_lines = line_result[0][0]
    ture_lines = line_result[1][0]
    for line in false_lines:
        cv2.line(img, (int(line[0]), int(line[1])), (int(line[2]), int(line[3])), color=(255,0,0),thickness=thickness)
    for line in ture_lines:
        cv2.line(img, (int(line[0]), int(line[1])), (int(line[2]), int(line[3])), color=(0,255,0),thickness=thickness)
    cv2.imwrite(out_file, img)

# 根据colab状态设置device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 选择模型对应的配置文件
# config = './configs/material/faster_rcnn_swim_fpn_1x_material_refine.py'
config = './configs/material/cascade_rcnn_swin_fpn_1x_material_refine.py'
# config = './configs/material/mask_rcnn_swin_fpn_1x_material_refine.py'
# config = './configs/icdar/cascade_rcnn_swin_fpn_1x_icdar2017_refine.py'
# config = './configs/icdar/cascade_mask_rcnn_r50_fpn_1x_icdar2019_refine.py'
# 选择下载好的checkpoint
# checkpoint = r'D:\jeffry\Repos\sources\mmdetection\work_dirs\faster_rcnn_swim_fpn_1x_material_refine\epoch_12.pth'
checkpoint = r'D:\jeffry\Repos\sources\mmdetection\work_dirs\cascade_rcnn_swin_fpn_1x_material_refine/epoch_11.pth'
# checkpoint = r'D:\jeffry\Repos\sources\mmdetection\work_dirs\mask_rcnn_swin_fpn_1x_material_refine/epoch_12.pth'
# checkpoint = r'D:\jeffry\Repos\sources\mmdetection\work_dirs\cascade_mask_rcnn_r50_fpn_1x_icdar2019_refine/epoch_12.pth'
# checkpoint = r'D:\jeffry\Repos\sources\mmdetection\work_dirs\cascade_rcnn_swin_fpn_1x_icdar2017_refine/epoch_11.pth'
# 载入推理图片
img_folder = r"D:\jeffry\Repos\sources\mmdetection\data\material_papers_e\img"
# img_folder = r"D:\jeffry\Repos\sources\mmdetection\data\ICDAR2019_cTDaR\test\TRACKA"
# img_folder = r"D:\jeffry\Repos\sources\mmdetection\data\ICDAR2017_POD\Images\TestSet"

# coco_data = COCO(annotation_file=r"D:\jeffry\Repos\sources\mmdetection\data\material_papers_e\ann\val.json")
coco_data = COCO(annotation_file=r"D:\jeffry\Repos\sources\mmdetection\data\material_papers_e\ann\coco_formate_ann\test.json")
# coco_data = COCO(annotation_file=r"D:\jeffry\Repos\sources\mmdetection\data\ICDAR2017_POD\Annotations\TestSet\test.json")
img_Ids = coco_data.getImgIds()
imgs = coco_data.loadImgs(img_Ids)
# 初始化模型
model = init_detector(config, checkpoint, device=device)

ocr_result_json = r"./ocrtools/baidu/material_results.json"
with open(ocr_result_json) as f:
    ocr_result = json.load(f)

dataset = 'material'
# m = 'Cascade RCNN'
m = 'baidu'

ori_output_folder = f"D:\\{dataset}\\{m}\\ori"
refine_output_folder = f"D:\\{dataset}\\{m}\\refine"
line_cls_output_folder = f"D:\\{dataset}\\{m}\\line_cls"
cmp_folder = f"D:\\{dataset}\\{m}\\cmp"

true_line = 0
false_line = 0
bad = 0
good = 0
complete = 0
cross = 0
ref_complete = 0
ref_cross = 0
total = 0

for img in imgs:
    result = inference_detector(model, os.path.join(img_folder, img['file_name']))


    if len(result) == 2:
        line_results = result[1]
        bbox_result = result[0]
    else:
        line_results = result[2]
        bbox_result = result[0]
    
    if img['file_name'] == '1-s2.0-S1044580310000902-main_3.png':
        print()
    
    # 直接从json导入ocr结果
    bbox_result = []
    for res in ocr_result['annotations']:
        if res['file_name'] == img['file_name']:
            bbox_result.append(res['bbox']+[0.99])
    if len(bbox_result) != 0:
        bbox_result = np.asarray(bbox_result)
    else:
        bbox_result.append([1,1,4,4,0.5])
        bbox_result = np.asarray(bbox_result)
    bbox_result = [bbox_result]

    true_line += line_results[0][0].shape[0]
    false_line += line_results[1][0].shape[0]
    # 保存线段分类的结果
    draw_line_cls(
        img = os.path.join(img_folder, img['file_name']),
        line_result = line_results,
        out_file= os.path.join(line_cls_output_folder, img['file_name'])
    )
    # 保存原bboxes结果
    model.show_result(
        os.path.join(img_folder, img['file_name']),
        bbox_result,
        score_thr=0.9,
        show=True,
        win_name='result',
        bbox_color=None,
        text_color=(200, 200, 200),
        thickness = 1,
        font_size = 5,
        mask_color=None,
        out_file= os.path.join(ori_output_folder, img['file_name'])
        )

    # 保存修正后bboxes结果
    refine_bbox_result = refine_bboxes(copy.deepcopy(bbox_result), line_results)
    model.show_result(
        os.path.join(img_folder, img['file_name']),
        refine_bbox_result,
        score_thr=0.9,
        show=True,
        win_name='result',
        bbox_color=None,
        text_color=(200, 200, 200),
        thickness = 1,
        font_size = 5,
        mask_color=None,
        out_file= os.path.join(refine_output_folder, img['file_name'])
        )

    ###################################################
    table_bboxes = bbox_result[0]
    ref_bboxes = refine_bbox_result[0]
    ids = np.where(table_bboxes[:,4]>=0.9)
    table_bboxes = table_bboxes[ids]
    ref_bboxes = ref_bboxes[ids]

    anns = coco_data.loadAnns(coco_data.getAnnIds(imgIds=img['id'],catIds=1)) # table

    anns_i = coco_data.loadAnns(coco_data.getAnnIds(imgIds=img['id'],catIds=3))
    anns_o = coco_data.loadAnns(coco_data.getAnnIds(imgIds=img['id'],catIds=4))
    anns.sort(key=lambda e: e['group'])
    anns_i.sort(key=lambda e: e['group'])
    anns_o.sort(key=lambda e: e['group'])
    gt_bboxes = [[ann['bbox'][0], ann['bbox'][1], ann['bbox'][0]+ann['bbox'][2], ann['bbox'][1]+ann['bbox'][3]] for ann in anns]
    gt_inners = [[ann['bbox'][0], ann['bbox'][1], ann['bbox'][0]+ann['bbox'][2], ann['bbox'][1]+ann['bbox'][3]] for ann in anns_i]
    gt_outters = [[ann['bbox'][0], ann['bbox'][1], ann['bbox'][0]+ann['bbox'][2], ann['bbox'][1]+ann['bbox'][3]] for ann in anns_o]

    # img_array = cv2.imread(os.path.join(img_folder, img['file_name']), cv2.IMREAD_COLOR)
    
    ori_img_array = cv2.imread(os.path.join(ori_output_folder, img['file_name']), cv2.IMREAD_COLOR)
    refine_img_array = cv2.imread(os.path.join(refine_output_folder, img['file_name']), cv2.IMREAD_COLOR)
    cmp_img_array = cv2.imread(os.path.join(img_folder, img['file_name']), cv2.IMREAD_COLOR)

    for table_bbox, ref_bbox in zip(table_bboxes, ref_bboxes):

        flag = False
        flag_ref = False
        for inner, outter in zip(gt_inners, gt_outters):
            if(in_area(table_bbox[:4], inner, outter)):
                flag = True
            if(in_area(ref_bbox[:4], inner, outter)):
                flag_ref = True
                pass
            pass
        if flag == True:
            complete += 1
        else:
            cross += 1
        if flag_ref == True:
            ref_complete += 1
        else:
            ref_cross += 1

        iou = round(compu_iou(table_bbox, gt_bboxes), 2)
        riou = round(compu_iou(ref_bbox, gt_bboxes), 2)
        color = (255,0,0)
        if riou < 0.9:
            color = (0,255,0)
        if iou>=0.9 and riou <0.9:
            bad += 1
        if iou<0.9 and riou>=0.9:
            good +=1
        text_pos = (int(table_bbox[0]),int(table_bbox[3]))
        ref_text_pos = (int(ref_bbox[0]),int(ref_bbox[3]))

        # img_array = cv2.putText(img_array, f"IOU={iou}->{riou}",text_pos,0, 0.5,color,1)
        
        ori_img_array = cv2.putText(ori_img_array, f"IOU={iou}",text_pos,0, 0.5,color,1)
        ori_img_array = cv2.putText(ori_img_array, f"{flag}",(int(table_bbox[0]),int(table_bbox[1])),0, 0.5,color,1)
        refine_img_array = cv2.putText(refine_img_array, f"IOU={riou}",ref_text_pos,0, 0.5,color,1)
        refine_img_array = cv2.putText(refine_img_array, f"{flag_ref}",(int(ref_bbox[0]),int(ref_bbox[1])),0, 0.5,color,1)
        cv2.rectangle(cmp_img_array,(int(table_bbox[0]),int(table_bbox[1])), (int(table_bbox[2]),int(table_bbox[3])),(255,0,0))
        cv2.rectangle(cmp_img_array,(int(ref_bbox[0]),int(ref_bbox[1])), (int(ref_bbox[2]),int(ref_bbox[3])),(0,255,0))

    # cv2.imwrite(os.path.join(cmp_folder, img['file_name']), img_array)
    
    for gt_box in gt_bboxes:
        total += 1
        cv2.rectangle(cmp_img_array,(int(gt_box[0]),int(gt_box[1])), (int(gt_box[2]),int(gt_box[3])),(0,0,255))

    cv2.imwrite(os.path.join(cmp_folder, img['file_name']), cmp_img_array)
    cv2.imwrite(os.path.join(ori_output_folder, img['file_name']), ori_img_array)
    cv2.imwrite(os.path.join(refine_output_folder, img['file_name']), refine_img_array)
    #######################################################

# print(f"bad={bad},good={good}\n")
print(f"total={total}")
print(f"complete={complete}, cross={cross}")
print(f"ref_complete={ref_complete}, ref_cross={ref_cross}")
