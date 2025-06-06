#基于感知哈希算法的以图搜图
import glob

import cv2
import numpy as np
import matplotlib.pyplot as plt
#定义计算哈希值函数
def get_Hash(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    dst=(8,8)
    re_img=cv2.resize(gray,dst)
    mean=np.mean(re_img)
    r=(re_img>mean).astype(int)
    result=r.flatten()
    return result
#计算哈希值之间的汉明距离
def hamming(x1,x2):
    xor=cv2.bitwise_xor(x1,x2)
    result=np.sum(xor)
    return result
img=cv2.imread('apple.jpg',1)
hash=get_Hash(img)
print('目标图像的哈希值为：',hash)
#计算某一文件夹下所有图片的哈希值
images=[]
EXTS='jpg','jpeg','png','tif','bmp','gif'
for exts in EXTS:
    images.extend(glob.glob('image/*.%s'%exts)) #glob.glob()用于匹配路径 ().extend() 将找到的所有符合的放入序列中
seq=[]
for f in images:
    I=cv2.imread(f)
    seq.append((f,get_Hash(I)))
print('所有图片的哈希值为：',seq)
distance=[]
for x in seq:
    distance.append((hamming(hash,x[1]),x[0]))
    s=sorted(distance)
print('排序后的汉明距离为：',s)
#展示出图库中与目标图像最相似的三幅图像
img1=cv2.imread(s[0] [1])
img2=cv2.imread(s[1] [1])
img3=cv2.imread(s[2] [1])
plt.figure('result')
plt.subplot(141),plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)),plt.axis('off')
plt.subplot(142),plt.imshow(cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)),plt.axis('off')
plt.subplot(143),plt.imshow(cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)),plt.axis('off')
plt.subplot(144),plt.imshow(cv2.cvtColor(img3,cv2.COLOR_BGR2RGB)),plt.axis('off')
plt.show()

