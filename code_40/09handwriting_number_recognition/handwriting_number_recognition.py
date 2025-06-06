#手写数字识别
import cv2
import glob
#计算匹配值的函数
def getMatch(template,image):
    tem_gray=cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
    ret,tem_binary=cv2.threshold(tem_gray,0,255,cv2.THRESH_OTSU)
    img_gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    ret,img_binary=cv2.threshold(img_gray,0,255,cv2.THRESH_OTSU)
    h,w=img_gray.shape
    re_tem=cv2.resize(tem_binary,(w,h))
    result=cv2.matchTemplate(re_tem,img_binary,cv2.TM_CCOEFF)
    return result[0][0]

img=cv2.imread('9.bmp',1)
template_dir=[]
for i in range(10):
    template_dir.extend(glob.glob('image/'+str(i)+'/*.*'))
matchValue=[]
for j in template_dir:
    template=cv2.imread(j)
    value=getMatch(img,template)
    matchValue.append(
        value
    )
print('所有的匹配值为：',matchValue)
bestMatch=max(matchValue)
index=matchValue.index(bestMatch)
number=int(index/10)
print('该手写数字是：',number)
