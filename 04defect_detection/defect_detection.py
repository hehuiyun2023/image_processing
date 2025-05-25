#实现物体的缺陷检测
import os
import cv2
import numpy as np
new_dir='E:/IMG_learning/code_40/04defect_detection'
os.chdir(new_dir)
img=cv2.imread('./pill.jpg',1)
cv2.imshow('original',img)
#预处理
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret1,binary=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow('threshold',binary)
kernel=cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
opening1=cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
cv2.imshow('opening1',opening1)
#确定前景图像
dist_transform=cv2.distanceTransform(opening1,cv2.DIST_L2,3)
ret2,fore=cv2.threshold(dist_transform,0.3*dist_transform.max(),255,0)
cv2.imshow('fore',fore)
#去噪处理
kernel1=np.ones(shape=(3,3),dtype=np.uint8)
opening2=cv2.morphologyEx(fore,cv2.MORPH_OPEN,kernel1,iterations=1)
cv2.imshow('opening2',opening2)
#提取轮廓
opening2=np.array(opening2,np.uint8)
contours,hierarchy=cv2.findContours(opening2,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
#缺陷检测
count=0
font=cv2.FONT_HERSHEY_PLAIN
for cnt in contours:
    (x,y),radius=cv2.minEnclosingCircle(cnt)#确定最小包围圆形
    center=(int(x),int(y))
    r=int(radius)
    circle_img=cv2.circle(opening2,center,r,(255,255,255),1)
    area=cv2.contourArea(cnt)
    circle_area=3.14*r**2
    if area/circle_area>=0.5:
        img=cv2.putText(img,'OK',center,font,1,(255,255,255),2)
    else:
        img=cv2.putText(img,'Bad',center,font,1,(255,255,255),2)
    count+=1
img=cv2.putText(img,'sum='+str(count),(20,30),font,1,(255,0,0,),2)
cv2.imshow('result',img)
cv2.waitKey()
cv2.destroyAllWindows()



