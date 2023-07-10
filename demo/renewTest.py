import os
import json
import random
from pycocotools.coco import COCO
import copy

coco_ann_folder = r'./data/material_papers_e/ann'
lableme_ann_folder = r'./data/material_papers_e/ann/lableme_formate'
test_ann_file = os.path.join(coco_ann_folder, 'test.json')

if __name__ == '__main__':
    with open(test_ann_file, 'r') as f:
        dataset = json.load(f)
        assert type(dataset)==dict, 'annotation file format {} not supported'.format(type(dataset))
    coco = COCO(test_ann_file)
    img_ids = coco.getImgIds()
    img_infos = coco.loadImgs(img_ids)

    annotations = []
    for img_info in img_infos:
        file_name = os.path.splitext(img_info['file_name'])[0]
        with open(os.path.join(lableme_ann_folder, file_name+'.json')) as f:
            json_file = json.load(f)
            for shape in json_file['shapes']:
                if shape['label'] == 'table':
                    category_id = 1
                elif shape['label'] == 'head':
                    category_id = 2
                elif shape['label'] == 'inner':
                    category_id = 3
                elif shape['label'] == 'outter':
                    category_id = 4
                else:
                    category_id = 0
                x_min, y_min = [int(x) for x in shape['points'][0]]
                x_max, y_max = [int(x) for x in shape['points'][1]]
                annotation = {
                    "id": len(annotations),
                    "image_id": img_info['id'],
                    "category_id": category_id,
                    "segmentation": [[x_min, y_min, x_min, y_max, x_max, y_max, x_max, y_min]],
                    "bbox": [x_min, y_min, x_max-x_min, y_max-y_min],
                    "area": (x_max-x_min)*(y_max-y_min),
                    "group": shape['group_id'],
                    "iscrowd": 0
                }
                annotations.append(annotation)
    dataset['annotations'] = annotations
    with open(os.path.join(coco_ann_folder, 'renew_test.json'), 'w') as f:
        jsondata = json.dumps(dataset,indent=4,separators=(',', ': '))
        f.write(jsondata)
        print(f'输出在目录 {coco_ann_folder}')
    print('done!')
