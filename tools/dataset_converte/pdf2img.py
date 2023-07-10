import os, sys
import datetime
import fitz
import uuid
import argparse

# requirement
# pip install fitz
# pip install PyMuPDF


# dataset_root 为工作目录下的相对目录， pdf_folder, img_folder 都假定为root下面的文件夹
# zoom_fac 为缩放因子影响分辨率 1 --> 612x792 dip=96 (1.33333333 --> 816x1056) (2 --> 1224x1584)
class PDF2Img():
    def __init__(self, dataset_root: str, pdf_folder = 'pdf', img_folder = 'img', zoom_fac = 1) -> None:
        pdf_folder = os.path.join(dataset_root, pdf_folder)
        img_folder = os.path.join(dataset_root, img_folder)
        assert os.path.exists(dataset_root)
        assert os.path.exists(pdf_folder)
        assert os.path.exists(img_folder)
        
        self.dataset_root = dataset_root
        self.pdf_folder = pdf_folder
        self.img_folder = img_folder

        self.zoom_fac = zoom_fac
        # os.mkdir(self.pdf_folder)
        # os.mkdir(self.img_folder)
        self.done = 0   # converted pdf number
        self.converted_pages = 0

    # img_folder 假定为 root 下的文件夹
    def set_img_folder(self, img_folder: str) -> None: 
        img_folder = os.path.join(self.dataset_root, img_folder)
        assert os.path.exists(img_folder)
        self.img_folder = img_folder
    
    def set_zoom(self, zoom) -> None:
        self.zoom_fac = zoom
    
    def stat_convert(self):
        print(f'program starat: convert pdf from {self.pdf_folder} to {self.img_folder}')
        fileNameList = os.listdir(self.pdf_folder)
        for fileName in fileNameList:
            if os.path.splitext(fileName)[1]=='.pdf':
                pdfPath = os.path.join(self.pdf_folder, fileName)
                self.pyMuPDF_fitz(pdfPath, self.img_folder)
        print(f'program done: ')
        print(f'program starat: convert pdf from {self.pdf_folder} to {self.img_folder}, zoom factors set {self.zoom_fac}')
        print(f'total convert {self.done} pdf files, output {self.converted_pages} images')


    # pdfPath: pdf文件路径 imageFolder: 图像输出文件
    def pyMuPDF_fitz(self, pdfPath:str, imageFolder:str):        
        # startTime = datetime.datetime.now() # 开始时间

        pdfDoc = fitz.open(pdfPath)
        title = os.path.basename(pdfPath).split('.pdf')[0]
        pages = 0
        # title = uuid.uuid4().hex
        for pg in range(pdfDoc.page_count):
            page = pdfDoc[pg]
            rotate = int(0)
            # 默认大小为 1 --> 612x792 dip=96 (1.33333333 --> 816x1056) (2 --> 1224x1584)
            zoom_x = self.zoom_fac 
            zoom_y = self.zoom_fac
            mat = fitz.Matrix(zoom_x, zoom_y).prerotate(rotate)
            pix = page.get_pixmap(matrix=mat, alpha=False)

            pix.save(os.path.join(imageFolder, f'{title}_{pg}.png'))
            pages += 1
            # pix.writePNG(os.path.join(imageFolder, f'{title}_{pg}.png'))   #将图片写入指定的文件夹内

        # endTime_pdf2img = datetime.datetime.now()#结束时间
        self.done += 1
        self.converted_pages += pages
        print(f'[{self.done}]: {title}.pdf converted, total {pages} pages')

def parse_args():
    parser = argparse.ArgumentParser(description='convert pdf files to imgs.')
    parser.add_argument('root', help='dataset root folder')
    parser.add_argument('--pdf', default='pdf', help='pdf file folder')
    parser.add_argument('--img', default='img', help='output img folder')
    parser.add_argument('--zoom', default=1.0, help='determin output resolution (1 --> 612x792 dip=96) (1.33333333 --> 816x1056) (2 --> 1224x1584)')
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    converter = PDF2Img(args.root, args.pdf, args.img, args.zoom)
    converter.stat_convert()


import re
if __name__ == "__main__":

    main()
