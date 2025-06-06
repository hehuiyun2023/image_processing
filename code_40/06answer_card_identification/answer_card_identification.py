#整张答题卡答案识别
# 主要包括六步：1、图像预处理；2、答题卡预处理；3、筛选出所有选项；4、将选项按照题目分组；5、处理每一道题目的选项；6、结果显示。

import cv2
import numpy as np
from scipy.spatial import distance as dist
#step1:图像预处理
img=cv2.imread('F:/PythonProject/Images_Process/ComputeVersion_40/06answer_card_identification/b.jpg',1)
#cv2.imshow('original',img)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gaussian=cv2.GaussianBlur(gray,(5,5),0)
canny_edges=cv2.Canny(gaussian,50,200)
#ret,binary=cv2.threshold(gaussian,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
# cv2.imshow('binary',binary)
#step2：答题卡预处理
def myWrapPersective(image,pts):
    #确定四个顶点具体的位置
    xsorted = pts[pts[:, 0].argsort()]  # 按照x轴从左到右排序，效果相同 xsorted=pts[np.argsort(pts[:,0]),:]
    left=xsorted[:2,:]#左侧两个顶点
    right=xsorted[2:,:]#右侧两个顶点
    left=left[left[:,1].argsort()]#left=left[np.argsort(left[:,1]),:]
    (tl,bl)=left
    # D=dist.cdist(tl[np.newaxis],right,"euclidean")[0]
    # (br,tr)=right[np.argsort(D)[::-1],:] #[::-1]反转索引顺序,从远到近排序
    #或者用下面这个的代码找到右上和右下
    right=right[np.argsort(right[:,1]),:]
    (tr,br)=right
    src=np.array([tl,tr,br,bl],dtype="float32")
    #测试顶点位置是否正确
    srcx=np.array([tl,tr,br,bl],dtype="int32")
    #print('看看顶点在哪个位置:\n',srcx)
    test=image.copy()
    cv2.polylines(test,[srcx],True,(255,255,255),8) #白色边
    #cv2.imshow("test",test)
    widthA=np.sqrt((bl[0]-br[0])**2+(bl[1]-br[1])**2)
    widthB=np.sqrt((tl[0]-tr[0])**2+(tl[1]-tr[1])**2)
    maxWidth=max(int(widthB),int(widthA))
    heightA=np.sqrt((tl[0]-bl[0])**2+(tl[1]-bl[1])**2)
    heightB=np.sqrt((tr[0]-br[0])**2+(tr[1]-br[1])**2)
    maxHeight=max(int(heightB),int(heightA))
    dst=np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0,maxHeight-1]],dtype="float32")
    M=cv2.getPerspectiveTransform(src,dst) #src是原始图形四个顶点，dst是目标图像四个顶点，M是变换矩阵
    wraped=cv2.warpPerspective(image,M,(maxWidth,maxHeight))#透视变换，将图像image经过变换矩阵转换为大小为(maxWidth,maxHeight)的目标图像
    return wraped
cnts,h=cv2.findContours(canny_edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
sorted_cnts=sorted(cnts,key=cv2.contourArea,reverse=False)#找出答题卡最大轮廓以及最大轮廓的四个顶点
for c in sorted_cnts:
    epsilon=0.01*cv2.arcLength(c,True)
    approx=cv2.approxPolyDP(c,epsilon,True)
    if len(approx)==4:
        wrapped_img=myWrapPersective(img,approx.reshape(4,2))
        wrapped_gray=myWrapPersective(gray,approx.reshape(4,2))

#cv2.imshow('wraped_img',wrapped_img)
 #steps3:筛选出所有选项
ret,binary=cv2.threshold(wrapped_gray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
#cv2.imshow('binary',binary)
cnts1,h1=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
print('在校正后的答题卡中，共找到',len(cnts1),'个轮廓')
options=[]
wrap1=wrapped_img.copy()
wrap2=wrapped_img.copy()

for (ci,c) in enumerate(cnts1):
    x,y,w,h=cv2.boundingRect(c)
    ar=w/float(h)
    if w>=25 and h>=25 and ar>=0.6 and ar<=1.3:
        cv2.putText(wrap1,str(ci),(x-1,y-5),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),1)
        options.append(c)

#cv2.imshow('random_wraped',wrap1)
print('符合条件的轮廓个数为：',len(options))
# step4、将选项按照题目分组 and # step5、处理每一道题目的选项
boundingBoxes=[cv2.boundingRect(c) for c in options]
(options,boundingBoxes)=zip(*sorted(zip(options,boundingBoxes),key=lambda b:b[1][1],reverse=False)) #从上到下

answer_dict = {0: "A", 1: "B", 2: "C", 3: "D"}
answer =  {0: "B", 1: "C", 2:"A", 3: "C", 4: "D"}
right_num=0
for (ni,n) in enumerate(np.arange(0,len(options),4)):
    boundingBox=[cv2.boundingRect(c) for c in options[n:n+4]]
    (option,boundingBox)=zip(*sorted(zip(options[n:n + 4],boundingBox),key=lambda  b:b[1][0],reverse=False))#从左到右
    voxel_num = []
    for (oi,o) in enumerate(option):

        mask = np.zeros(shape=wrapped_gray.shape, dtype=np.uint8)
        x, y, w, h = cv2.boundingRect(o)
        cv2.putText(wrap2, str(n + oi), (x - 1, y - 5), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
        cv2.drawContours(mask,[o], -1,255, -1)
        mask = cv2.bitwise_and(binary, binary, mask=mask)
        #cv2.imshow('mask' + str(n + oi), mask)
        total = cv2.countNonZero(mask)  # 计算mask里面的白色像素个数，白色像素个数最多的是考生的答案
        voxel_num.append((total,oi))

    voxel_num = sorted(voxel_num, key=lambda b: b[0], reverse=True)
    choice_num = voxel_num[0][1]
    choice = answer_dict.get(choice_num)


    if choice == answer.get(ni):
        color = (0, 255, 0)#绿色
        text = 'T'
        print('第', str(ni+1), '题正确答案是:', answer.get(ni))
        print('第',str(ni+1),'题该生做出的选项是:',choice,'回答正确！')
        right_num+=1
    else:
        color = (0, 0, 255)#红色
        text = 'F'
        print('第', str(ni+1), '题正确答案是:', answer.get(ni))
        print('第', str(ni+1), '题该生做出的选项是:', choice, '回答错误！')
    cv2.putText(wrapped_img, text, (x-1, y-5), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

    #cv2.imshow('mask' + str(ni), mask)
s1 = "total: " + str(len(answer)) + ""
s2 = "right: " + str(right_num)
s3 = "score: " + str(right_num*1.0/len(answer)*100)+""
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(wrapped_img, s1 + "  " + s2+"  "+s3, (10, 30), font, 0.5, (0, 0, 255), 2)
#cv2.imshow("sorted_wrapped",wrap2)
cv2.imshow('final_img',wrapped_img)
cv2.waitKey()
cv2.destroyAllWindows()

