from cProfile import label
import json
import os
import abc
from re import X
import cv2
from cv2 import randShuffle
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='convert annotation formate to coco')
    parser.add_argument('origin', help='the origin annotation type')
    parser.add_argument('to', help='the object annotation type')
    parser.add_argument('input', help='origin annotation path folder')
    parser.add_argument('output', help='annotation output folser')
    parser.add_argument('img', help='images folder')
    parser.add_argument('--file-name', default='table', help='output file name if output is a single file')

    args = parser.parse_args()
    return args


class BUILDCOCO():
    def __init__(self, imgFolder: str, annFolder: str) -> None:
        self.imgFolder = imgFolder
        self.annFolder = annFolder
        self.info = self.buildInfo()
        self.license = self.buildLicense()
        self.categories = self.buildCategories()
        self.images = self.buildImages()
        self.outputAnn = self.buildAnn()

    def findImageId(self, imgname: str) -> int:
        for image in self.images:
            if(imgname == image['file_name']):
                return image['id']
        # 找不到文件，引发异常
        raise RuntimeError(f"can not find the file: {imgname}")

    def jsontext(self) -> dict:
        return {
            "info": self.info,
            "images": self.images,
            "annotations": self.outputAnn,
            "categories": self.categories,
            "licenses": self.license,
        }

    
    def buildLicense(self) -> list:
        return [
            {
                "url": "https://creativecommons.org/licenses/by-nc-nd/4.0/",
                "id": 1,
                "name": "Attribution-NonCommercial-NoDerivs License"
            }
        ]
    
    def buildCategories(self) -> list:
        return [
            {
                "id": 1, 
                "name": "table", 
                "supercategory": "table"
            },
            {
                "id": 2,
                "name": "head",
                "supercategory": "head"
            },
            {
                "id": 3,
                "name": "inner",
                "supercategory": "table"
            },
            {
                "id": 4,
                "name": "outter",
                "supercategory": "table"
            },
            {
                "id": 5,
                "name": "cell",
                "supercategory": "cell"
            }
        ]
    
    def buildImages(self) -> list:
        img_filenames = [x for x in os.listdir(self.imgFolder) if os.path.splitext(x)[1] in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG', '.TIFF']]
        images = []
        for index, img_name in enumerate(img_filenames):
            cv_img = cv2.imread((os.path.join(self.imgFolder, img_name)))
            image = {
                'id': index,
                'file_name': img_name,
                'height': list(cv_img.shape)[0],
                'width': list(cv_img.shape)[1],
                'license': 1
            }
            images.append(image)
        return images

    @abc.abstractclassmethod
    def buildInfo(self) -> dict:
        """
        {
            "year": 2021,
            "contributor": "KMUST ST Group",
            "date_created": "20121/10/06",
            "version": "1.0",
            "url": "",
            "description": "None"
        }
        """

    @abc.abstractclassmethod
    def buildAnn(self) -> list:
        """
        需要重写该方法
        单个annotation格式为:
        {
            "category_id": 1, 
            "area": 59780, 
            "iscrowd": 0, 
            "segmentation": [[90, 106, 90, 246, 517, 246, 517, 106]], 
            "id": 123, 
            "image_id": 97, 
            "bbox": [90, 106, 427, 140]
        }
        """

class ICDAR2017_2COCO(BUILDCOCO):
    """
        把ICDAR2017的xml文件转化为COCO数据格式
    """
    def buildInfo(self) -> dict:
        return {
            "year": 2021,
            "contributor": "KMUST ST Group",
            "date_created": "20121/10/06",
            "version": "1.0",
            "url": "",
            "description": "icdar"
        }

    def buildImages(self) -> list:
        import re
        img_filenames = [x for x in os.listdir(self.imgFolder) if os.path.splitext(x)[1] in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG', '.TIFF']]
        images = []
        for index, img_name in enumerate(img_filenames):
            cv_img = cv2.imread((os.path.join(self.imgFolder, img_name)))
            image = {
                'id': index,
                'file_name': img_name,
                'height': list(cv_img.shape)[0],
                'width': list(cv_img.shape)[1],
                'license': 1
            }
            images.append(image)
        return images

    def buildAnn(self) -> list:
        import xml.etree.ElementTree as ET
        import re
        xml_filenames = [x for x in os.listdir(self.annFolder) if os.path.splitext(x)[1] == '.xml']
        # 选择电子文件
        # modern_xml_filenames = [x for x in xml_filenames]
        annotations = []
        for filename in xml_filenames:
            tree = ET.parse(os.path.join(self.annFolder, filename))
            root = tree.getroot()
            for table in root:
                points = table.find('Coords').get('points')
                # 空白字符为分隔符
                points = points.split()
                x_min, y_min = [int(x) for x in points[0].split(',')]
                x_max, y_max = [int(x) for x in points[2].split(',')]
                table_x_min, table_y_min = x_min, y_min
                width = x_max - x_min
                heigth = y_max - y_min

                annotation = {
                        "id": len(annotations),
                        "image_id": self.findImageId(root.get('filename')),
                        "category_id": 1,
                        "segmentation": [[x_min, y_min, x_min, y_max, x_max, y_max, x_max, y_min]],
                        "bbox": [x_min, y_min, width, heigth],
                        "area": (x_max-x_min)*(y_max-y_min),
                        "iscrowd": 0
                }
                annotations.append(annotation)

                # cell 坐标
                for cell in table.iter('cell'):
                    points = cell.find('Coords').get('points')
                    points = points.split()
                    x_min, y_min = [int(x) for x in points[0].split(',')]
                    x_max, y_max = [int(x) for x in points[2].split(',')]
                    # 裁剪后
                    # x_min = table_x_min - x_min
                    # y_min = table_y_min - y_min
                    # x_max = table_x_min - x_max
                    # y_max = table_y_min = y_max
                    width = x_max - x_min
                    heigth = y_max - y_min
                    annotation = {
                            "id": len(annotations),
                            "image_id": self.findImageId(root.get('filename')),
                            "category_id": 5,
                            "segmentation": [[x_min, y_min, x_min, y_max, x_max, y_max, x_max, y_min]],
                            "bbox": [x_min, y_min, width, heigth],
                            "area": (x_max-x_min)*(y_max-y_min),
                            "iscrowd": 0
                    }
                    annotations.append(annotation)
        
        return annotations

class ICDAR2017_2COCO_STR(ICDAR2017_2COCO):
    def buildAnn(self) -> list:
        import xml.etree.ElementTree as ET
        import re
        xml_filenames = [x for x in os.listdir(self.annFolder) if os.path.splitext(x)[1] == '.xml']
        # 选择电子文件
        # modern_xml_filenames = [x for x in xml_filenames]
        annotations = []
        for filename in xml_filenames:
            tree = ET.parse(os.path.join(self.annFolder, filename))
            root = tree.getroot()
            for i, table in enumerate(root):
                points = table.find('Coords').get('points')
                # 空白字符为分隔符
                points = points.split()
                x_min, y_min = [int(x) for x in points[0].split(',')]
                x_max, y_max = [int(x) for x in points[2].split(',')]
                table_x_min, table_y_min = x_min, y_min

                # annotation = {
                #         "id": len(annotations),
                #         "image_id": self.findImageId(root.get('filename')),
                #         "category_id": 1,
                #         "segmentation": [[x_min, y_min, x_min, y_max, x_max, y_max, x_max, y_min]],
                #         "bbox": [x_min, y_min, width, heigth],
                #         "area": (x_max-x_min)*(y_max-y_min),
                #         "iscrowd": 0
                # }
                # annotations.append(annotation)

                # cell 坐标
                for cell in table.iter('cell'):
                    points = cell.find('Coords').get('points')
                    points = points.split()
                    x_min, y_min = [int(x) for x in points[0].split(',')]
                    x_max, y_max = [int(x) for x in points[2].split(',')]
                    # 裁剪后
                    x_min = x_min - table_x_min
                    y_min = y_min - table_y_min
                    x_max = x_max - table_x_min
                    y_max = y_max - table_y_min
                    width = x_max - x_min
                    heigth = y_max - y_min
                    image_id = self.findImageId(filename.split('.')[0] + f'_{i}' + '.jpg')
                    annotation = {
                            "id": len(annotations),
                            "image_id": image_id,
                            "category_id": 5,
                            "segmentation": [[x_min, y_min, x_min, y_max, x_max, y_max, x_max, y_min]],
                            "bbox": [x_min, y_min, width, heigth],
                            "area": (x_max-x_min)*(y_max-y_min),
                            "iscrowd": 0
                    }
                    annotations.append(annotation)
        
        return annotations

class ICDAR2019_2COCO(BUILDCOCO):
    """
        把ICDAR2019的xml文件转化为COCO数据格式
    """
    def buildInfo(self) -> dict:
        return {
            "year": 2021,
            "contributor": "KMUST ST Group",
            "date_created": "20121/10/06",
            "version": "1.0",
            "url": "",
            "description": "icdar"
        }

    def buildImages(self) -> list:
        import re
        img_filenames = [x for x in os.listdir(self.imgFolder) if os.path.splitext(x)[1] in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG', '.TIFF']]
        img_filenames = [x for x in img_filenames if re.match('cTDaR_t1', os.path.splitext(x)[0])]
        images = []
        for index, img_name in enumerate(img_filenames):
            cv_img = cv2.imread((os.path.join(self.imgFolder, img_name)))
            image = {
                'id': index,
                'file_name': img_name,
                'height': list(cv_img.shape)[0],
                'width': list(cv_img.shape)[1],
                'license': 1
            }
            images.append(image)
        return images

    def buildAnn(self) -> list:
        import xml.etree.ElementTree as ET
        import re
        xml_filenames = [x for x in os.listdir(self.annFolder) if os.path.splitext(x)[1] == '.xml']
        # 选择电子文件
        modern_xml_filenames = [x for x in xml_filenames if re.match('cTDaR_t1', os.path.splitext(x)[0])]
        # modern_xml_filenames = [x for x in xml_filenames]
        annotations = []
        for filename in modern_xml_filenames:
            tree = ET.parse(os.path.join(self.annFolder, filename))
            root = tree.getroot()
            for table in root:
                points = table.find('Coords').get('points')
                # 空白字符为分隔符
                points = points.split()
                x_min, y_min = [int(x) for x in points[0].split(',')]
                x_max, y_max = [int(x) for x in points[2].split(',')]
                width = x_max - x_min
                heigth = y_max - y_min

                annotation = {
                        "id": len(annotations),
                        "image_id": self.findImageId(root.get('filename')),
                        "category_id": 1,
                        "segmentation": [[x_min, y_min, x_min, y_max, x_max, y_max, x_max, y_min]],
                        "bbox": [x_min, y_min, width, heigth],
                        "area": (x_max-x_min)*(y_max-y_min),
                        "iscrowd": 0
                }
                annotations.append(annotation)
        
        return annotations


class LABELME2COCO(BUILDCOCO):
    """
        把labelme的单个json文件转化为COCO数据格式
    """
    def buildInfo(self) -> dict:
        return {
            "year": 2021,
            "contributor": "KMUST ST Group",
            "date_created": "20121/10/06",
            "version": "1.0",
            "url": "",
            "description": "material"
        }

    def buildAnn(self) -> list:
        json_filenames = [x for x in os.listdir(self.annFolder) if os.path.splitext(x)[1] == '.json']
        annotations = []
        for filename in json_filenames:
            with open(os.path.join(self.annFolder, filename)) as f:
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
                    group_id = shape['group_id']

                    annotation = {
                        "id": len(annotations),
                        "image_id": self.findImageId(os.path.split(json_file['imagePath'])[-1]),
                        "category_id": category_id,
                        "group": group_id,
                        "segmentation": [[x_min, y_min, x_min, y_max, x_max, y_max, x_max, y_min]],
                        "bbox": [x_min, y_min, x_max-x_min, y_max-y_min],
                        "area": (x_max-x_min)*(y_max-y_min),
                        "iscrowd": 0
                    }
                    annotations.append(annotation)
            
        return annotations


def main():
    # 这里假定目标格式为 coco 如有变化需要修改
    args = parse_args()
    assert os.path.exists(args.input)
    assert os.path.exists(args.output)
    assert os.path.exists(args.img)
    coco_data = None
    if args.origin == 'labelme' and args.to == 'coco':
        coco_data = LABELME2COCO(args.img, args.input)
    
    if args.origin == 'ICDAR2017' and args.to == 'coco':
        coco_data = ICDAR2017_2COCO(args.img, args.input)

    if args.origin == 'ICDAR2017' and args.to == 'coco + STR':
        coco_data = ICDAR2017_2COCO_STR(args.img, args.input)
    
    assert coco_data is not None
    # 获取json数据
    jsontext = coco_data.jsontext() #<------这是个字典

    # 写json文件
    with open(os.path.join(args.output, f'{args.file_name}.json'), 'w') as f:
        jsondata = json.dumps(jsontext,indent=4,separators=(',', ': '))
        f.write(jsondata)
        print(f'output folder: {args.output}, file: {args.file_name}')

if __name__ == '__main__':
    main()
    