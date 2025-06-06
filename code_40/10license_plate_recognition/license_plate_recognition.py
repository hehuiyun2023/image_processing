# 车牌识别:1、提取车牌；2，分割车牌；3，识别车牌。
import cv2
import glob
import numpy as np


# 找到车牌所在位置
def getPlate(image):
    img = image.copy()
    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # sobel 算子进行边缘检测
    sobelX = cv2.Sobel(image, cv2.CV_16S, 1, 0)
    absx = cv2.convertScaleAbs(sobelX)
    image = absx
    ret, image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel1)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 19))
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel2)
    image = cv2.medianBlur(image, 15)  # 中值滤波
    contours, h = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnts in contours:
        x, y, w, h = cv2.boundingRect(cnts)
        if w > (h * 3):
            plate = img[y:y + h, x:x + w]
    return plate


# 对车牌进行预处理
def preprocessing(image):
    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ret, image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    #image=cv2.erode(image,kernel)
    image = cv2.dilate(image, kernel,iterations=1)
    return image


# 对车牌内的字符进行分割
def splitPlate(image):
    contours, h = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    words = []
    for cnts in contours:
        rect = cv2.boundingRect(cnts)  # rect:x/y/w/h:word[3]
        words.append(rect)
    words = sorted(words, key=lambda b: b[0], reverse=False)  # 按照从左到右进行排序
    plateChars = []
    for word in words:
        if (word[2] > 3) and (word[3] > word[2] * 1.3) and (word[3] < word[2] * 8):
            plateChar = image[word[1]:word[1] + word[3], word[0]:word[0] + word[2]]
            plateChars.append(plateChar)
    print('字符个数为：', len(plateChars))
    return plateChars

#
# 车牌字典
templateDict = {0:'0',1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',
            10:'A',11:'B',12:'C',13:'D',14:'E',15:'F',16:'G',17:'H',
            18:'J',19:'K',20:'L',21:'M',22:'N',23:'P',24:'Q',25:'R',
            26:'S',27:'T',28:'U',29:'V',30:'W',31:'X',32:'Y',33:'Z',
            34:'京',35:'津',36:'冀',37:'晋',38:'蒙',39:'辽',40:'吉',41:'黑',
            42:'沪',43:'苏',44:'浙',45:'皖',46:'闽',47:'赣',48:'鲁',49:'豫',
            50:'鄂',51:'湘',52:'粤',53:'桂',54:'琼',55:'渝',56:'川',57:'贵',
            58:'云',59:'藏',60:'陕',61:'甘',62:'青',63:'宁',64:'新',
            65:'港',66:'澳',67:'台'}
# 号牌第一位是汉字，为省、自治区、直辖市的简称：
 #号牌第二位是字母，是发牌机关代号（也可以理解为市区代号），除直辖市外，其他省份或自治区的号牌没有字母I和O。
#号牌 第三位至第七位，是车牌序号。



# 读取所有模板图片的路径
def getTempalte():
    c = []
    for i in range(0,68):
        word = []
        word.extend(glob.glob('template/' + templateDict.get(i) + '/*.*'))
        c.append(word)
        # print(templateDict.get(i)+':\n',word)

    return c


# 计算模板图片路径所对应的图片和目标图片之间的匹配值
def getMatch(image, template):
    template_img = cv2.imdecode(np.fromfile(template, dtype=np.uint8), 1)
    template_img = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
    ret, template_img = cv2.threshold(template_img, 0, 255, cv2.THRESH_OTSU)
    h, w = image.shape
    template_img = cv2.resize(template_img, (w, h))
    result = cv2.matchTemplate(image, template_img, cv2.TM_CCOEFF)
    return result[0][0]


# 对车牌内所有字符进行识别
def getBestMatch(plateChars, templates):
    results = []
    for platechar in plateChars:
        bestMatch = []
        for template in templates:
            match = []
            for word in template:
                values = getMatch(platechar, word)
                match.append(values)
            bestMatch.append(max(match))

        # print(bestMatch)
        # print(max(bestMatch))
        index = bestMatch.index(max(bestMatch))
        r = templateDict[index]
        results.append(r)
    return results


# 主程序
image = cv2.imread('gua.jpg', 1)
cv2.imshow('original', image)
plate = getPlate(image)
cv2.imshow('plate', plate)
plate_pre = preprocessing(plate)
cv2.imshow('plate_preprocessing', plate_pre)
plateChars = splitPlate(plate_pre)
for i, im in enumerate(plateChars):
    cv2.imshow('char' + str(i), im)
templates = getTempalte()
results = getBestMatch(plateChars, templates)
print(results)
results = "".join(results)
print('识别结果为：', results)
cv2.waitKey()
cv2.destroyAllWindows()
