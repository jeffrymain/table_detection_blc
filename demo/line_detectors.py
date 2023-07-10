import cv2
import os
import numpy as np
import json



img_name = '1-s2.0-S0167577X01004189-main_6.png'
img = cv2.imread(os.path.join('./demo', img_name))
width = img.shape[1]

green = (0, 255, 0)

# # hough
# houghimg = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
# houghimg = cv2.GaussianBlur(houghimg,(3,3),0)
# edges = cv2.Canny(houghimg, 150, 350, apertureSize = 5)
# hough_lines = cv2.HoughLinesP(edges,1,np.pi/180,threshold=100,minLineLength=int(width/5),maxLineGap=5)

# # fld
# gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# fld = cv2.ximgproc.createFastLineDetector()
# fld_lines = fld.detect(gray_img)


# canny line
with open(r"D:\jeffry\Repos\sources\mmdetection\data\material_papers_e\ann\cannylines_result.json", 'r') as f:
    canny = json.load(f)["line_detected"]
for img_inf in canny:
    if img_inf['file_name'] == img_name:
        canny_lines = np.asarray(img_inf['line'])
        canny_lines = canny_lines[:,0:4].astype(int)
        canny_lines = canny_lines[:,np.newaxis,:]
        break

# for line in hough_lines:
#     line = line[0]
#     hough_img = cv2.line(img, (line[0], line[1]), (line[2], line[3]), green)

# for line in fld_lines:
#     line = line[0].astype(int)
#     fld_img = cv2.line(img, (line[0], line[1]), (line[2], line[3]), green)

for line in canny_lines:
    line = line[0].astype(int)
    canny_img = cv2.line(img, (line[0], line[1]), (line[2], line[3]), green)
    pass

# cv2.imwrite('./hough_line.jpg', hough_img)
# cv2.imwrite('./fld_line.jpg', fld_img)
cv2.imwrite('./canny_line.jpg', canny_img)

pass