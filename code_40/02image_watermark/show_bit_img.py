#展示图像的不同位平面
import cv2
import numpy as np
img=cv2.imread('img.jpg',0)
h,w=img.shape
mask=np.zeros(shape=(h,w,8),dtype=np.uint8)
for i in range(8):
    mask[:,:,i]=2**i
result_img=np.zeros(shape=(h,w,8),dtype=np.uint8)
for i in range(8):
    result_img[:,:,i]=cv2.bitwise_and(img,mask[:,:,i])
    result_img[result_img[:,:,i]>0]=255
    cv2.imshow(str(i),result_img[:,:,i])
cv2.waitKey()
cv2.destroyAllWindows()



