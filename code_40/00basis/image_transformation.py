import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('F:\PythonProject\photo\input.jpg')
height, width = img.shape[:2]
# 向沿X平移20
mat_translation = np.float32([[1, 0, 20],[0, 1, 0]])
dst1 = cv2.warpAffine(img, mat_translation, (width, height))
cv2.imshow('x label translation', dst1)
#1水平镜像0垂直镜像-1对角镜像
dst2 = cv2.flip(img , -1)
cv2.imshow('flip', dst2)
#图像缩放
dst_height=int(height*0.5)
dst_width=int(width*0.5)
dst3=cv2.resize(img,(dst_width,dst_height),0,0)
cv2.imshow('change_size',dst3)
#图像的转置
dst4=cv2.transpose(img)
cv2.imshow('transposed_image',dst4)
#图像旋转
mat_rotate=cv2.getRotationMatrix2D((0,0),45,0.5)
dst5=cv2.warpAffine(img,mat_rotate,(width,height))
cv2.imshow('rotated_image',dst5)
#图像剪切
dst6=img[100:200,50:150]
cv2.imshow('jianqie_image',dst6)
#图像二维离散傅里叶变换
gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
f=np.fft.fft2(gray_img)
fshift=np.fft.fftshift(f)
fimg=np.log(np.abs(fshift))
plt.subplot(121), plt.imshow(gray_img,'gray'),plt.title('Input Image')
plt.subplot(122), plt.imshow(fimg,'gray'),plt.title('FFT Image')
cv2.waitKey(0)
