import cv2
import numpy as np
# 读取图像，默认彩色图像
img = cv2.imread('F:\PythonProject\photo\input.jpg')
# 显示图像
cv2.imshow('original_image.jpg', img)

# 彩色图像转换为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray_image', gray)

# 灰度图像转化为二值图像
ret, dst = cv2.threshold(gray, 127, 255, 0)  # 127为阈值，255为图像数据最大值
cv2.imshow('binary_image', dst)
cv2.waitKey(0)
# 保存图像
cv2.imwrite('F:\PythonProject\photo\output.jpg', gray)
# 视频文件读取
cap = cv2.VideoCapture('F:\PythonProject\photo\video.mp4')
if cap.isOpened():
    while True:
        ret, img = cap.read()
        if ret:
            cv2.imshow('image', img)
            cv2.waitKey(0)
        else:
            break
else:
    print('视频打开失败！')



