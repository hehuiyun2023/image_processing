# 线性变换
import cv2
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# 扩展动态范围
def grayHist(img):
    h, w = img.reshape[: 2]
    pixelsequence = img.reshape([h * w, ])
    numberbins = 256

#非运算
def get_img_reverse(img):
    dst = cv.bitwise_not(img)


img = cv.imread('F:\PythonProject\photo\input.jpg', 0)
dst = cv.bitwise_not(img)
out = 2.0 * img
out[out > 255] = 0
out = np.around(out)
out = out.astype(np.uint8)
result = np.hstack((out, dst))  # 合并函数
cv.imshow('original', img)
cv.imshow('linear_gray_enhancement', out)
cv.imshow('reversed_img', dst)
cv.imshow('merged_image', result)
cv.waitKey(0)

# 非线性变换
import cv2
import numpy as np
import matplotlib.pyplot as plt


def log_plot(c):
    x = np.arange(0, 256, 0.01)
    y = c * np.log(1 + x)
    plt.plot(x, y, 'r', linewidth=1)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文标签
    plt.title('对数变换函数曲线')
    plt.xlim(0, 255), plt.ylim(0, 255)
    plt.show()


def log(c, img):
    output = c * np.log(1.0 + img)
    output = np.uint8(output + 0.5)
    return output


img = cv2.imread('F:\PythonProject\photo\input.jpg', 0)

log_plot(40)
output = log(40, img)
cv2.imshow('original', img)
cv2.imshow('log_changed_img', output)
cv2.waitKey(0)
