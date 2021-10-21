import scipy.io as sio
import cv2
import numpy as np
import math

imagePath = '/home/saptarshi/PythonCode/AdvanceLSTM/MatlabScripts/images/NavtechImages/'
imageSavePath = '/home/saptarshi/PythonCode/AdvanceLSTM/MatlabScripts/images/BoxImages/'
fileNames = ["000001", '000002',"000003","000004","000005","000006","000007","000008","000009","000010","000011","000012","000013","000014"]
imageType = '.png'


def DrawAngledRec(oldx0, oldy0, width, height, angle, img, color):

    x0 = oldx0 - 576
    y0 = 576 - oldy0

    _angle = angle * math.pi / 180.0
    b = math.cos(_angle) * 0.5
    a = math.sin(_angle) * 0.5
    pt0 = (int(x0 - a * height - b * width) + 576,
           int(y0 + b * height - a * width) + 576)
    pt1 = (int(x0 + a * height - b * width) + 576,
           int(y0 - b * height - a * width) + 576)
    pt2 = (int(pt0[0] - (2 * x0)), int(pt0[1] - (2 * y0)))
    pt3 = (int(pt1[0] - (2 * x0)), int(pt1[1] - 2 * y0))

    cv2.line(img, pt0, pt1, color, 3)
    cv2.line(img, pt1, pt2, color, 3)
    cv2.line(img, pt2, pt3, color, 3)
    cv2.line(img, pt3, pt0, color, 3)

    return img



boundingBoxes = sio.loadmat('/home/saptarshi/PythonCode/AdvanceLSTM/MatlabScripts/matlab.mat')
allFrames = boundingBoxes['allFrmaesBoxes'][0]
for idx in range(0,len(allFrames),25):
    radarImg = cv2.imread(imagePath+fileNames[idx]+imageType)
    allBoxes = allFrames[idx:idx+25]
    for jdx in range(0,len(allBoxes),5):
        x,y,w,h,ang = allBoxes[jdx:jdx+5]
        radarImg = DrawAngledRec(x,y,w,h,ang,radarImg,[255,0,0])
        cv2.imshow('test', radarImg)
        cv2.waitKey(0)
print(boundingBoxes)