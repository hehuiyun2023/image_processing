#用于判断单道题目的答案是否正确
import cv2
import numpy as np
# 答案和选项的初始化
answer_dict = {0: "A", 1: "B", 2: "C", 3: "D"}
answer = "C"
#图像预处理
img=cv2.imread('F:/PythonProject/Images_Process/ComputeVersion_40/06answer_card_identification/xiaogang.jpg',1)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
filtered_img=cv2.GaussianBlur(gray,(5,5),0)
ret,binary=cv2.threshold(filtered_img,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
cv2.imshow('binary',binary)
#获取图像轮廓
cnts,h=cv2.findContours(binary.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) #cv2.RETR_EXTERNAL用来检测图像的外轮廓，不关注内轮廓
# 将轮廓按照从左到右排列，方便后续处理
boundingBoxes = [cv2.boundingRect(c) for c in cnts]
(cnts,boundingBoxes)=zip(*sorted(zip(cnts,boundingBoxes),key=lambda b:b[1][0],reverse=False) )
options=[]
for (j,c) in enumerate(cnts):
    mask=np.zeros(shape=gray.shape,dtype=np.uint8)
    cv2.drawContours(mask,[c],-1,255,-1) #通过循环，将每一个轮廓放入单独的mask里面
    mask=cv2.bitwise_and(binary,binary,mask=mask)
    cv2.imshow('mask'+str(j),mask)
    total=cv2.countNonZero(mask)#计算mask里面的白色像素个数，白色像素个数最多的是考生的答案
    options.append((total,j))

print('排序前：',options)
options=sorted(options,key=lambda b:b[0],reverse=True)
print('排序后：',options)
choice_num=options[0][1]
choice=answer_dict.get(choice_num)
print('该生的做出的选项是：',choice)
if choice==answer:
    color=(0,255,0)
    text='right'
    print('该生回答正确')
else:
    color=(0,0,255)
    text='wrong'
    print('该生回答错误！')
cv2.putText(img,text,(0,80),cv2.FONT_HERSHEY_PLAIN,2,color,2)
cv2.imshow('results',img)
cv2.waitKey()
cv2.destroyAllWindows()
