import os
from re import M
from unittest import result
from matplotlib import image
import numpy as np
import torch
from PIL import Image
import cv2
from coco import COCO

class MaterialDataset(object):
    def __init__(self, root, annFile=None, transforms=None):
        # load all image files, sorting them to 
        # ensure that they are aligned
        self.root = root
        self.transforms = transforms
        self.type = type
        self.img_root = os.path.join(root, 'img')
        # 如果annFile不为空，则为自定义标记
        if annFile == None:
            annFile = os.path.join(root, 'ann', 'table.json')
        self.coco = COCO(annFile)
        
        self.cats = self.coco.loadCats(self.coco.getCatIds())
        self.catIds = self.coco.getCatIds(catNms=['table'])
        self.imgIds = self.coco.getImgIds(catIds=self.catIds)

        pass

        
    def __getitem__(self, idx):
        # load images and masks
        imgIds = self.imgIds[idx]
        img_info = self.coco.loadImgs(imgIds)[0]

        img_path = os.path.join(self.img_root, img_info['file_name'])
        img = Image.open(img_path).convert('RGB')

        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        annIds = self.coco.getAnnIds(imgIds=img_info['id'], catIds=self.catIds, iscrowd=None)
        anns = self.coco.loadAnns(annIds)

        # 生成mask
        masks = np.zeros(shape=(len(anns),img_info['height'],img_info['width']))
        for index, ann in enumerate(anns): 
            m = self.coco.annToMask(ann)    # m的维度是高x宽
            m = m == [[[1]]]
            # print(np.sum(m))
            masks[index] = m

        # 给每个mask生成 bounding box coordinates
        num_objs = len(anns)
        boxes = []
        for i in range(num_objs):
            xmin = anns[i]['bbox'][0]
            ymin = anns[i]['bbox'][1]
            xmax = xmin + anns[i]['bbox'][2]
            ymax = ymin + anns[i]['bbox'][3]
            boxes.append(np.asarray([xmin, ymin, xmax, ymax],dtype=np.int64))
        
        # 把所有东西转成tensor格式
        boxes = np.array(boxes)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # 只有一个种类--表格
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([imgIds])
        area = torch.as_tensor([ann['area'] for ann in anns], dtype=torch.float32)
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        houghimg = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        houghimg = cv2.GaussianBlur(houghimg,(3,3),0)
        edges = cv2.Canny(houghimg, 50, 150, apertureSize = 3)
        lines = cv2.HoughLinesP(edges,1,np.pi/180,int(img_info['width']/4),minLineLength=int(img_info['width']/4),maxLineGap=3)

        # houghimg = np.zeros((img_info['height'], img_info['width'],3), np.uint8)
        # houghimg.fill(255)
        # houghimg = cv2.cvtColor(houghimg, cv2.COLOR_RGB2BGR)
        # if lines is not None:
        #     for line in lines:
        #         for x1,y1,x2,y2 in line:
        #             cv2.line(img,(x1,y1),(x2,y2),(0,0,0),2)

        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        cv2.imshow(f'idx={idx},img_index={imgIds}', img)
        cv2.waitKey()

        if lines is not None:
            lines = np.squeeze(lines, axis=1)
            lines = torch.as_tensor(lines, dtype=torch.float32)
        else:
            lines = torch.tensor([0,0,img_info['width'],0], dtype=torch.float32)
            lines = lines.resize(1,4)
        target['lines'] = lines
        # houghimg = Image.fromarray(cv2.cvtColor(houghimg, cv2.COLOR_BGR2RGB))

        if self.transforms is not None:
            img, target = self.transforms(img, target)
            # houghimg, _ = self.transforms(houghimg, None)

        return img, target

    def __len__(self):
        return len(self.imgIds)
        pass
    
    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))

if __name__ == '__main__':
    root = 'D://wolftail//repos//datasets//material_papers'
    train_set = MaterialDataset(root=root)
    # train_set[78369]
    # train_set[59648]
    train_set[33]
    for obj in train_set:
        pass
