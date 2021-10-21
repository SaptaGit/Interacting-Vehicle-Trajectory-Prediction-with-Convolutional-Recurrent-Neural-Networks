from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
#from cStringIO import StringIO
from io import StringIO
import urllib
import csv
from pyproj import Proj, transform
import io
import cv2
import sys
import math

Image.MAX_IMAGE_PIXELS = 198036602
# print(Image.MAX_IMAGE_PIXELS)
# pil_img = Image.open('/home/saptarshi/Documents/I-80-Emeryville-CA/aerial-ortho-photos/emeryville1.tif')
# dtype = {'F': np.float32, 'L': np.uint8}[pil_img.mode]
# np_img = np.array(pil_img.getdata(), dtype=dtype)
# w, h = pil_img.size
# reshaped = np.reshape(np_img, (w,h))
# np_img.shape = (h, w, np_img.size // (w * h))
# print(pil_img.size)
# print(np_img.shape)
# imageSect = reshaped[3000:4000,14000:16000]
# print(imageSect.shape)
# print(imageSect.dtype)
# im = Image.fromarray(np.uint8(imageSect), mode='L')
# #plt.imshow(imageSect)
# #plt.show()
# im.show()

# url = 'http://maps.googleapis.com/maps/api/staticmap?maptype=satellite&center=' + str(37.8466) + ' ' + str(-122.2987) + '&size=640x640&zoom=19&key=AIzaSyBuYwhrUegFSvy2UaLDXd52CxiuXlMsLm4&sensor=false'
# #print(url)
# buffer = StringIO(urllib.request.urlopen(url).read())
# image = Image.open(buffer).convert('RGB')
# open_cv_image = numpy.array(image)
# open_cv_image = open_cv_image[:, :, ::-1].copy()
# #cv2.imshow('Test', open_cv_image)
# #cv2.waitKey(0)



# tiffstack = Image.open('/home/saptarshi/Documents/I-80-Emeryville-CA/aerial-ortho-photos/emeryville1.tif')
# tiffstack.load()
# tiffstack.seek(0)
# ar = np.array(tiffstack)
# extractArr = ar[12700:13500, 1400:2400]
# #extractArr = ar[250:254, 30:32]
# plt.imshow(extractArr)
# plt.show()
# im = Image.fromarray(extractArr.astype('uint8'), 'L')
# im.save('test.png')
# #imgTest = Image.fromarray(ar)
# #imgTest.show()
# print(tiffstack.n_frames)

loadFileName = 'vehicle4.csv'
loadFile = open(loadFileName, 'r')
loadReader = csv.reader(loadFile)
loadDataset = []
for loadRow in loadReader:
    loadDataset.append([loadRow[0],loadRow[1],loadRow[2],loadRow[3],loadRow[4],loadRow[5],loadRow[6],loadRow[7]])

loadDataset.pop(0)
sortedList = sorted(loadDataset, key=lambda x: (float(x[0]), float(x[1])))
datasetArray = np.array(sortedList, dtype=np.float)

xPosition = datasetArray[:,6]
yPosition = datasetArray[:,7]
fgbg = cv2.createBackgroundSubtractorMOG2()
for idx in range(0,len(xPosition)):
    inProj = Proj(init='epsg:2227', preserve_units = True)
    outProj = Proj(init='epsg:4326')
    x1,y1 = xPosition[idx],yPosition[idx]
    x2,y2 = transform(inProj,outProj,x1,y1)
    position = str(y2) + ',' + str(x2)
    urlparams = urllib.parse.urlencode({'center': position,
                                    'zoom': str(20),
                                    'size': '640x640',
                                    'maptype': 'satellite',
                                    'sensor': 'true',
                                    'key':'AIzaSyBuYwhrUegFSvy2UaLDXd52CxiuXlMsLm4'})
    url = 'http://maps.google.com/maps/api/staticmap?' + urlparams


    f = urllib.request.urlopen(url).read()
    im=Image.open(io.BytesIO(f)).convert('RGB')
    opencvImage = np.array(im)
    fgmask = fgbg.apply(opencvImage)
    res = cv2.bitwise_and(opencvImage,opencvImage,mask = fgmask)
    #masked = np.multiply(opencvImage,fgmask)
    cv2.imshow('frame',opencvImage)
    #cv2.imshow('test.png',opencvImage)
    cv2.waitKey(1)


# plt.scatter(xPosition,yPosition)
# plt.show()



