#实现图像中的物体计数
import cv2
img=cv2.imread('./code_40/03object_counting/count.jpg',1)
#预处理
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,binary=cv2.threshold(gray,150,255,cv2.THRESH_BINARY_INV)
kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
erosion=cv2.erode(binary,kernel,iterations=4)
dilation=cv2.dilate(erosion,kernel,iterations=3)
gaussian=cv2.GaussianBlur(dilation,(3,3),0)#高斯滤波
#找到所有轮廓
contours,hierarchy=cv2.findContours(gaussian,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
#筛选出合适的轮廓
contourOK=[]
for i in contours:
    if cv2.contourArea(i)>30:
        contourOK.append(i)
#绘制轮廓
draw=cv2.drawContours(img,contourOK,-1,(0,255,0),1)
#计算每一个轮廓的质心，并绘制序号
for i,j in zip(contourOK,range(len(contourOK))):
    M=cv2.moments(i)
    cx=int(M['m10']/M['m00'])
    cy=int(M['m01']/M['m00'])
    cv2.putText(draw,str(j),(cx,cy),cv2.FONT_HERSHEY_PLAIN,1.5,(0,0,255),2)
cv2.imshow('gaussian',gaussian)
cv2.imshow('contours',draw)
cv2.waitKey()
cv2.destroyAllWindows()