#实现手势0和手势1的区分
import cv2
def reg(x):
    x=cv2.cvtColor(x,cv2.COLOR_BGR2GRAY)
    #ret,binary=cv2.threshold(gray,127,255,cv2.THRESH_OTSU)
    contours,hierarchy=cv2.findContours(x,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnt=max(contours,key=lambda x:cv2.contourArea(x))
    cntarea=cv2.contourArea(cnt)
    hull=cv2.convexHull(cnt)
    hullarea=cv2.contourArea(hull)
    area_ratio=cntarea/hullarea
    if area_ratio>0.8:
        result='finger:0'
    else :
        result='finger:1'
    return result
x=cv2.imread('F:/PythonProject/Images_Process/ComputeVersion_40/05gesture_recognition/one.jpg',1)
y=cv2.imread('F:/PythonProject/Images_Process/ComputeVersion_40/05gesture_recognition/zero.jpg',1)
xtext=reg(x)
ytext=reg(y)
cv2.putText(x,xtext,(0,80),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),3)
cv2.putText(y,ytext,(0,80),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),3)
cv2.imshow('one',x)
cv2.imshow('zero',y)
cv2.waitKey()
cv2.destroyAllWindows()
