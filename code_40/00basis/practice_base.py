import cv2
import numpy as np
img = cv2.imread("../img.jpg", 1)
kernel1=np.zeros((3,3),np.uint8)
kernel2=np.array([[255,255,255],[255,255,255],[255,255,255]],dtype=np.uint8)
erode_img=cv2.erode(img,kernel2)
dilate_img=cv2.dilate(erode_img,kernel2)
cv2.imshow('before',img)
cv2.imshow("erode",erode_img)
cv2.imshow('dilate',dilate_img)
cv2.waitKey()
cv2.destroyAllWindows()

s_img=cv2.blur(img,ksize=(3,3))
gs_img=cv2.GaussianBlur(img,ksize=(3,3),sigmaX=0,sigmaY=0)
gs_img2=cv2.GaussianBlur(img,ksize=(3,3),sigmaX=0.1,sigmaY=0.2)
ms_img=cv2.medianBlur(img,ksize=3)

r=cv2.imwrite('../result.jpg', img)
b,g,r=cv2.split(img)
bgr=cv2.merge([b,g,r])
rgb=cv2.merge([r,g,b])
cv2.imshow('right',bgr)
cv2.imshow('wrong',rgb)

a=np.array([0,2,4],dtype='uint8')

example=np.zeros((8,8),dtype=np.uint8)
cv2.imshow("one",example)
example[0:3,:]=255
print('修改后的图像为：\n',example)
cv2.imshow("two",example)


# 怀旧滤镜
def vintage_filter(img):
    b, g, r = cv2.split(img)
    new_r = 0.393*r + 0.769*g + 0.189*b
    new_g = 0.349*r + 0.686*g + 0.168*b
    new_b = 0.272*r + 0.534*g + 0.131*b
    filtered = cv2.merge([new_b, new_g, new_r]).clip(0, 255).astype(np.uint8)
    return filtered
# bgr蓝色成分
def blue_filter(img):
    b, g, r = cv2.split(img)
    zeros = np.zeros(img.shape[:2], dtype="uint8")  # 创建与image相同大小的零矩阵
    filtered = cv2.merge([b, zeros, zeros])
    return filtered

# BGR颜色空间转换为灰度空间
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 图像缩放和裁剪
new_width = 480
new_height =480
resized = cv2.resize(img, (new_width, new_height))
crop_y_start = 100
crop_y_end = 300
crop_x_start = 100
crop_x_end = 300
cropped = img[crop_y_start:crop_y_end, crop_x_start:crop_x_end]


# 显示结果
vintage_img = vintage_filter(img)
blue_img = blue_filter(img)
cv2.imshow("Original", img)
cv2.imshow("Vintage", vintage_img)
cv2.imshow("Blue", blue_img)
cv2.imshow("Gray", gray)
cv2.imshow("Resized", resized)
cv2.imshow("Cropped", cropped)
cv2.waitKey(0)

# 保存图像
cv2.imwrite("F:\PythonProject\photo\output.jpg", vintage_img)




