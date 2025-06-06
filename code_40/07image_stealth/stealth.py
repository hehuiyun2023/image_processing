#将图片中红色区域的人隐身
import cv2
import numpy as np

img=cv2.imread('fore.jpg',1)
background=cv2.imread('back.jpg',1)#图片背景
cv2.imshow('original',img)
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
#识别红色区域
lower_red=np.array([0,120,70])
upper_red=np.array([10,255,255])
mask1=cv2.inRange(hsv,lower_red,upper_red)
lower_red=np.array([170,120,70])
upper_red=np.array([180,255,255])
mask2=cv2.inRange(hsv,lower_red,upper_red)
mask=mask1+mask2
cv2.imshow('mask',mask)
mask_inv=cv2.bitwise_not(mask)
cv2.imshow('mask_inv',mask_inv)
back_mask=cv2.bitwise_and(background,background,mask=mask)
img_mask=cv2.bitwise_and(img,img,mask=mask_inv)
stealth=cv2.add(img_mask,back_mask)
cv2.imshow('stealth',stealth)
cv2.waitKey()
cv2.destroyAllWindows()
