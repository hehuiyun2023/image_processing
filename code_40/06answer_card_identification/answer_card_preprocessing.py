#对答题卡进行倾斜校正和裁边处理
import cv2
import numpy as np
from scipy.spatial import distance as dist
#自定义透视函数,pts是逼近多边形的四个顶点
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
    cv2.imshow("test",test)
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
#图像预处理
img=cv2.imread('F:/PythonProject/Images_Process/ComputeVersion_40/06answer_card_identification/b.jpg',1)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gaussian=cv2.GaussianBlur(gray,(5,5),0)
canny_edges=cv2.Canny(gaussian,50,200)
#ret,binary=cv2.threshold(gaussian,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
cv2.imshow('edges',canny_edges)
# cv2.imshow('binary',binary)
#找出所有外部轮廓
cnts,h=cv2.findContours(canny_edges.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
contours_img=cv2.drawContours(img.copy(),cnts,-1,(0,0,255),2)
cv2.imshow("contours_img",contours_img)
print('找到的轮廓个数：',len(cnts))
list=sorted(cnts,key=cv2.contourArea,reverse=False )#按照轮廓面积排序
for c in list:
    # cv.approxPolyDP函数有三个参数，分别是：curve：输入多边形的轮廓。epsilon：逼近精度参数，表示逼近精度的界限。该参数是一个正数，其值越小则逼近程度越高。
    # 通常建议使用轮廓周长的一定比例来计算该参数，常见的比例因子为0.01。
    # closed：布尔值参数，表示输出的逼近多边形是否闭合。如果布尔值为True，则输出的多边形是封闭的。如果为False，则只返回线段。

    epsilon=0.01*cv2.arcLength(c,True) #arcLength 计算轮廓周长
    approx=cv2.approxPolyDP(c,epsilon,True)
    print('逼近多边形的顶点个数：',len(approx))
    if len(approx)==4:
        corrected_img=myWrapPersective(img,approx.reshape(4,2))
cv2.imshow("corrected_img",corrected_img)
cv2.waitKey()
cv2.imwrite('corrected_b.jpg',corrected_img)
cv2.destroyAllWindows()




