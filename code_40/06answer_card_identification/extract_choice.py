#用于提取整张答题卡的选项并按照题目分组
import cv2
import numpy as np

img=cv2.imread('F:/PythonProject/Images_Process/ComputeVersion_40/06answer_card_identification/corrected_b.jpg',1)
cv2.imshow('original',img)
# 图像预处理
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# gaussian=cv2.GaussianBlur(gray,(5,5),0)
ret,binary=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
cnts,h=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
print('共找到',len(cnts),'个轮廓')
options=[]
image=img.copy()
sorted_img=img.copy()
for (i,ci) in enumerate(cnts):
    x,y,w,h=cv2.boundingRect(ci)#获取轮廓的矩形包围框
    ar=w/float(h) #计算纵横比，w宽度，h高度
    if  w>=25 and h >=25 and ar>=0.6 and ar<=1.3:
        options.append(ci)
        cv2.putText(image,str(i),(x-1,y-5),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),1)
print('共找到',len(options),'个符合条件的轮廓')
cv2.drawContours(image,options,-1,(0,0,255),1)
cv2.imshow('random_image',image)#选项无序时的图像
#将选项按照从上到下、从左到右的排序
boundingBoxes=[cv2.boundingRect(c) for c in options]
(options,boundingBoxes)=zip(*sorted(zip(options,boundingBoxes),key=lambda b:b[1][1],reverse=False))#从上到下排序
for (tn,i) in enumerate(np.arange(0,len(options),4)):
    boundingBox=[cv2.boundingRect(c) for c in options[i:i+4]]
    (cnt,boundingBox)=zip(*sorted(zip(options[i:i+4],boundingBox),key=lambda b:b[1][0],reverse=False))#从左到右排序
    mask=np.zeros(shape=img.shape,dtype=np.uint8)
    for (n,ni) in enumerate(cnt):
        x,y,w,h=cv2.boundingRect(ni)
        cv2.putText(sorted_img,str(n+i),(x-1,y-5),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),1)
        cv2.drawContours(mask,[ni],-1,(255,255,255),-1)
        cv2.putText(mask,str(n),(x-1,y-5),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),2)
    cv2.imshow('mask'+str(tn),mask)
cv2.imshow("sorted_image",sorted_img)
cv2.waitKey()
cv2.destroyAllWindows()
