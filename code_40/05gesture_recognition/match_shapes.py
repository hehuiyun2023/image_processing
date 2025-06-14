#计算3个不同图像的匹配度
import cv2
o1=cv2.imread('o1.jpg',1)
o2=cv2.imread('o2.jpg',1)
o3=cv2.imread('o3.jpg',1)
gray1=cv2.cvtColor(o1,cv2.COLOR_BGR2GRAY)
gray2=cv2.cvtColor(o2,cv2.COLOR_BGR2GRAY)
gray3=cv2.cvtColor(o3,cv2.COLOR_BGR2GRAY)
ret1,binary1=cv2.threshold(gray1,127,255,cv2.THRESH_BINARY)
ret2,binary2=cv2.threshold(gray2,127,255,cv2.THRESH_BINARY)
ret3,binary3=cv2.threshold(gray3,127,255,cv2.THRESH_BINARY)
contours1,h1=cv2.findContours(binary1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
contours2,h2=cv2.findContours(binary2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
contours3,h3=cv2.findContours(binary3,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cnt1=contours1[0]
cnt2=contours2[0]
cnt3=contours3[0]
match1=cv2.matchShapes(cnt1,cnt1,1,0.0)
match2=cv2.matchShapes(cnt1,cnt2,1,0.0)
match3=cv2.matchShapes(cnt1,cnt3,1,0.0)
print(match1)
print(match2)
print(match3)
cv2.imshow('1',o1)
cv2.imshow('2',o2)
cv2.imshow('3',o3)
cv2.waitKey()
cv2.destroyAllWindows()