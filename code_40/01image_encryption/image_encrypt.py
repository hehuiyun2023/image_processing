#异或运算实现图像（整体或ROI)的加密与解密

#整体图像的加密和解密
import cv2
import numpy as np
img=cv2.imread("img.jpg",0)
h,w=img.shape
key=np.random.randint(0,256,size=[h,w],dtype=np.uint8)
encryption=cv2.bitwise_xor(img,key)#加密后的图像
decryption=cv2.bitwise_xor(encryption,key)#解密后的图像
cv2.imshow('image',img)
cv2.imshow('encrypted_image',encryption)
cv2.imshow('decrypted_image',decryption)
cv2.waitKey()
cv2.destroyAllWindows()

#面部打码与解码1
#面部信息img[200:480,420:640]
import cv2
import numpy as np
img=cv2.imread("img.jpg",0)
h,w=img.shape
mask=np.zeros(shape=(h,w),dtype=np.uint8)
mask[200:480,420:640]=1
lmask=1-mask
#将脸部打码
key=np.random.randint(0,256,size=(h,w),dtype=np.uint8)
image_key=cv2.bitwise_xor(img,key)#获得整个图像与秘钥加密后的图像
face_key=cv2.bitwise_and(image_key,mask*255)#获得只有面部加密的图像
image_lmask=cv2.bitwise_and(img,lmask*255)#将图像与lmask与运算，获得面部之外的信息
face_encryption=face_key+image_lmask#获得人像面部加密后的图像
cv2.imshow('face_encryption',face_encryption)
#将脸部解码
face_decryption=cv2.bitwise_xor(face_encryption,key)
face=cv2.bitwise_and(face_decryption,mask*255)
not_face=cv2.bitwise_and(face_encryption,lmask*255)
decrption=face+not_face
cv2.imshow('face_decryption',decrption)
cv2.waitKey()
cv2.destroyAllWindows()


#面部的打码与解码2（以ROI的方式）
import cv2
import numpy as np
img=cv2.imread('img.jpg',0)
h,w=img.shape
key=np.random.randint(0,256,size=(h,w),dtype=np.uint8)
dilraba_face=img[200:480,420:640] #设定ROI
cv2.imshow('ROI',dilraba_face)
img_key=cv2.bitwise_xor(img,key)
encryption_face=img_key[200:480,420:640]
cv2.imshow('encryption_face',encryption_face)
img[200:480,420:640]=encryption_face#此时的img已发生变化
cv2.imshow('encrypted_img',img)
decrption_img=cv2.bitwise_xor(img,key)
decrption_face=decrption_img[200:480,420:640]
cv2.imshow('decryption_face',decrption_face)
img[200:480,420:640]=decrption_face
cv2.imshow('decrypted_img',img)
cv2.waitKey()
cv2.destroyAllWindows()




