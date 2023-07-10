import argparse
import os


# 专门为mattab制作的删除无用图片程序
def parse_args():
    parser = argparse.ArgumentParser(description='delete imgs do not contain table')
    parser.add_argument('labelme', help='labelme annotation folder')
    parser.add_argument('img', help='img folder')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    assert os.path.exists(args.labelme)
    assert os.path.exists(args.img)
    img_names = [x for x in os.listdir(args.img) if os.path.splitext(x)[1] in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG', '.TIFF']]
    ann_names = [os.path.splitext(x)[0] for x in os.listdir(args.labelme) if os.path.splitext(x)[1] == '.json']
    
    for i, img_name in enumerate(img_names):
        if os.path.splitext(img_name)[0] not in ann_names:
            os.remove(os.path.join(args.img, img_name))
        print(f"[{i+1}] img have been scaned")

if __name__ == '__main__':
    main()