#实现所有手势识别0-5
import cv2
import numpy as np
import math
img=cv2.imread('F:/PythonProject/Images_Process/ComputeVersion_40/05gesture_recognition/test3.jpg',1)
#色彩空间转换和预处理
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# 使用自适应阈值替代全局阈值
ret,binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
cv2.imshow('binary',binary)
kernel=np.ones(shape=(3,3),dtype=np.uint8)
mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)  # 使用闭运算替代
cv2.imshow('filtered',mask)
#获取轮廓、轮廓面积、凸包、凸包面积
contours,hierarchy=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img,contours,-1,(255,0,0),1)
cnt=max(contours, key=lambda x:cv2.contourArea(x))
cnt_area=cv2.contourArea(cnt)
hull=cv2.convexHull(cnt)
hull_area=cv2.contourArea(hull)
area_ratio=cnt_area/hull_area #ratio大于0.9是0，小于的是数字1
#获取凸缺陷
hull=cv2.convexHull(cnt,returnPoints=False)
defects=cv2.convexityDefects(cnt,hull)
#凸缺陷处理
n=0
for i in range(defects.shape[0]):
    s,e,f,d=defects[i,0]#s 起点 e 终点 f 轮廓上距离凸包最远的点 d 最远点到凸包的近似距离
    start=tuple(cnt[s][0])
    end=tuple(cnt[e][0])
    far=tuple(cnt[f][0])
    a = math.dist(start, end)  # 更简洁的距离计算
    b = math.dist(far, start)
    c = math.dist(far, end)

    denominator = 2 * b * c
    if denominator == 0:
        continue  # 跳过无效计算

    cos_theta = (b ** 2 + c ** 2 - a ** 2) / denominator
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # 限制数值范围
    angle = math.degrees(math.acos(cos_theta))  # 直接使用math.degrees

    actual_d = d / 256.0  # 修正深度值
    if angle <= 90 and actual_d > 20:
        n += 1
        cv2.circle(img, far, 3, ( 255, 0,255), -1)
        cv2.line(img, start, end, (0, 255, 0), 2)

# 分类逻辑优化
if n == 0:
    result = 'finger:0' if area_ratio > 0.8 else 'finger:1'  # 调整阈值
elif n==1:
    result = 'finger:2'  # 假设 n+1 是手指数
elif n==2:
    result='finger:3'
elif n==3:
    result='finger:4'
elif n==4:
    result='finger:5'
else:
    result='wrong'

cv2.putText(img,result,(80,80),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),3)
cv2.imshow('result',img)
cv2.waitKey()
cv2.destroyAllWindows()