import os
import json
import random
from pycocotools.coco import COCO
import copy

# 拆分 coco数据集
def split_coco(annFile:str, val:float=0.2, test:float=0.2):
    with open(annFile, 'r') as f:
        dataset = json.load(f)
        assert type(dataset)==dict, 'annotation file format {} not supported'.format(type(dataset))
    coco = COCO(annFile)

    catIds = coco.getCatIds(catNms=['table'])
    imgIds = coco.getImgIds(catIds=catIds)

    spImgIds = {
        'train': [],
        'val': [],
        'test': []
    }
    numData = len(imgIds)

    # 划分图片
    # 测试图片
    innerIds = coco.getCatIds(catNms=['inner'])
    spImgIds['test'] = coco.getImgIds(catIds=innerIds)

    # 删除训练集
    for i in range(1, numData):
        if imgIds[numData-i] in spImgIds['test']:
            imgIds.pop(numData-i)
            numData -= 1
    

    # 训练与验证
    # for i in range(0, int(numData * test)):
    #     spImgIds['test'].append(imgIds.pop(random.randint(0, len(imgIds)-1)))
    for i in range(0, int(numData * val)):
        spImgIds['val'].append(imgIds.pop(random.randint(0, len(imgIds)-1)))
    spImgIds['train'] = imgIds

    # 划分标记
    spAnnIds = {}
    for key in spImgIds:
        spAnnIds[key] = coco.getAnnIds(spImgIds[key])
    
    # 建立划分后的数据集
    spDataSet = {}
    for key in spAnnIds:
        spDataSet[key] = copy.deepcopy(dataset)
        spDataSet[key]['annotations'] = coco.loadAnns(spAnnIds[key])
        spDataSet[key]['images'] = coco.loadImgs(spImgIds[key])
        pass

    return spDataSet['train'], spDataSet['val'], spDataSet['test']

if __name__ == '__main__':
    coco_ann_folder = r'D:\jeffry\Repos\sources\mmdetection\data\material_papers_e\ann\coco_formate_ann'
    coco_ann_file = r'D:\jeffry\Repos\sources\mmdetection\data\material_papers_e\ann\coco_formate_ann\table.json'
    data = {}
    data['train'], data['val'], data['test'] = split_coco(coco_ann_file)

    for key, value in data.items():
        js_test = json.dumps(value, indent=4,separators=(',', ': '))
        js_path = coco_ann_folder + f'\{key}.json'
        with open(js_path, 'w+') as f:
            f.write(js_test)
    pass
