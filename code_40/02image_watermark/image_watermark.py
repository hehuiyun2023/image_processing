#为图像添加水印
import cv2
import numpy as np
img=cv2.imread('img.jpg',0)
wm=cv2.imread('E:\IMG_learning\code_40\wm.jpg',0)
h,w=img.shape
resized_wm=cv2.resize(wm,(w,h))
#水印嵌入
T2=np.ones(shape=(h,w),dtype=np.uint8)#用于水印的提取
T1=255-T2#用于水印的嵌入
high_7=cv2.bitwise_and(T1,img)#保留图像的高7位
wm1=resized_wm.copy()
wm1[resized_wm>0]=1
img_wm=cv2.bitwise_or(high_7,wm1)
#水印提取
watermark=cv2.bitwise_and(img_wm,T2)
watermark1=watermark.copy()
watermark1[watermark>0]=255
cv2.imshow('watermark',resized_wm)
cv2.imshow('original',img)
cv2.imshow('add_wm',img_wm)
cv2.imshow('ext_wm',watermark1)
cv2.waitKey()
cv2.destroyAllWindows()


#可视化水印
import cv2
import numpy as np
img=cv2.imread('img.jpg',0)
wm=cv2.imread('E:\IMG_learning\code_40\wm.jpg',0)
h,w=img.shape
resized_wm=cv2.resize(wm,(w,h))
watermark_r=255-resized_wm
result1=img+watermark_r
result2=cv2.add(img,watermark_r)
result3=cv2.addWeighted(img,0.6,watermark_r,0.3,55)
result4=cv2.bitwise_or(img,resized_wm)#实现艺术字效果
cv2.imshow('wm_revise',watermark_r)
cv2.imshow('re1',result1)
cv2.imshow('re2',result2)
cv2.imshow('re3',result3)
cv2.imshow('re4',result4)
cv2.waitKey()
cv2.destroyAllWindows()