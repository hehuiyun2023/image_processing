#识别剪刀石头布
import cv2
def reg(x):
    o1 = cv2.imread('paper.jpg', 1)
    o2 = cv2.imread('rock.jpg', 1)
    o3 = cv2.imread('scissors.jpg', 1)
    gray1 = cv2.cvtColor(o1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(o2, cv2.COLOR_BGR2GRAY)
    gray3 = cv2.cvtColor(o3, cv2.COLOR_BGR2GRAY)
    x_gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    ret1, binary1 = cv2.threshold(gray1, 127, 255, cv2.THRESH_BINARY)
    ret2, binary2 = cv2.threshold(gray2, 127, 255, cv2.THRESH_BINARY)
    ret3, binary3 = cv2.threshold(gray3, 127, 255, cv2.THRESH_BINARY)
    ret4, x_binary = cv2.threshold(x_gray,127,255,cv2.THRESH_BINARY)
    contours1, h1 = cv2.findContours(binary1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours2, h2 = cv2.findContours(binary2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours3, h3 = cv2.findContours(binary3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x_contours,h4=cv2.findContours(x_binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnt1 = contours1[0]
    cnt2 = contours2[0]
    cnt3 = contours3[0]
    cnt_x=x_contours[0]
    match1 = cv2.matchShapes(cnt1, cnt_x, 1, 0.0)
    match2 = cv2.matchShapes(cnt2, cnt_x, 1, 0.0)
    match3 = cv2.matchShapes(cnt3, cnt_x, 1, 0.0)
    match=[match1,match2,match3]
    max_index=match.index(min(match))
    if max_index==0:
        result='paper'
        cv2.putText(x,result,(0,80),cv2.FONT_HERSHEY_PLAIN,2,(255,0,255),3)
    elif max_index==1:
        result='rock'
        cv2.putText(x, result, (0, 80), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3)
    else:
        result='scissors'
        cv2.putText(x, result, (0, 80), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3)
    return x

t1=cv2.imread('test1.jpg',1)
t2=cv2.imread('test2.jpg',1)
t3=cv2.imread('test3.jpg',1)
cv2.imshow('t1',reg(t1))
cv2.imshow('t2',reg(t2))
cv2.imshow('t3',reg(t3))
cv2.waitKey()
cv2.destroyAllWindows()
