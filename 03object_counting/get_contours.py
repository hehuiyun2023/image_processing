#基础知识：绘制质心、轮廓以及计算轮廓面积
import cv2
img=cv2.imread('E:\IMG_learning\code_40\cat3.jpg',1)
cv2.imshow('original',img)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,binary=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
contours,hierarchy=cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
n=len(contours)
for i in range(n):
    print("counter" + str(i) +"面积是：", cv2.contourArea(contours[i]))
    cv2.drawContours(img,contours,i,(0,0,255),2)
cv2.imshow('result1',img)
x=cv2.drawContours(img, contours, 0, (0, 0, 255), 2)#绘制轮廓
m00=cv2.moments(contours[0])['m00']
m01=cv2.moments(contours[0])['m01']
m10=cv2.moments(contours[0])['m10']
cx=int(m10/m00)#计算质心
cy=int(m01/m00)
cv2.putText(img,"center",(cx,cy),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),5)
cv2.imshow('result2',img)
cv2.waitKey()
cv2.destroyAllWindows()