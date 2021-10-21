import urllib
import urllib.request
from PIL import Image
from io import StringIO
import io
import cv2
from math import log, exp, tan, atan, pi, ceil
import numpy as np
import math
import csv
from pyproj import Proj, transform
import imutils
#Final code for map and cropped map as well***************************

#Bottom Right 37.8385, -122.2962  (South/East) (lat,lon)
# Top left 37.8466,-122.2987 (North/West) (lat,lon)


# a = 3041.3386
# b = 1650
# c = math.sqrt(pow(a,2) - pow(b,2))



feetPerPx = 156543.03392 * math.cos(37.84255 * math.pi / 180) / math.pow(2, 21) * 3.28084
pxPerFeet = 1/feetPerPx




EARTH_RADIUS = 6378137
EQUATOR_CIRCUMFERENCE = 2 * pi * EARTH_RADIUS
INITIAL_RESOLUTION = EQUATOR_CIRCUMFERENCE / 256.0
ORIGIN_SHIFT = EQUATOR_CIRCUMFERENCE / 2.0

def latlontopixels(lat, lon, zoom):
    mx = (lon * ORIGIN_SHIFT) / 180.0
    my = log(tan((90 + lat) * pi/360.0))/(pi/180.0)
    my = (my * ORIGIN_SHIFT) /180.0
    res = INITIAL_RESOLUTION / (2**zoom)
    px = (mx + ORIGIN_SHIFT) / res
    py = (my + ORIGIN_SHIFT) / res
    return px, py

def pixelstolatlon(px, py, zoom):
    res = INITIAL_RESOLUTION / (2**zoom)
    mx = px * res - ORIGIN_SHIFT
    my = py * res - ORIGIN_SHIFT
    lat = (my / ORIGIN_SHIFT) * 180.0
    lat = 180 / pi * (2*atan(exp(lat*pi/180.0)) - pi/2.0)
    lon = (mx / ORIGIN_SHIFT) * 180.0
    return lat, lon


# px1,py1 = latlontopixels(37.8385, -122.2962, 21)
# px2,py2 = latlontopixels(37.8466, -122.2962, 21)

# dpx = px1 - px2
# dpy = py1 - py2

# w = mapImage.shape[1]
# h = mapImage.shape[0]
# zoom = 21
# lat = -122.2962 - ((-122.2962 + 122.2987)/2)
# lng = 37.8385 + ((37.8466-37.8385)/2)

# def getPointLatLng(pointLng, pointLat):
#     parallelMultiplier = math.cos(lat * math.pi / 180)
#     degreesPerPixelX = 360 / math.pow(2, zoom + 8)
#     degreesPerPixelY = 360 / math.pow(2, zoom + 8) * parallelMultiplier
#     y = ((lat-pointLat)/degreesPerPixelY) + (h/2)
#     x = ((pointLng-lng)/degreesPerPixelX) + (w/2)

#     return (x, y)

# ############################################

# # a neighbourhood in Lajeado, Brazil:

# # upperleft =  '-29.44,-52.0'  
# # lowerright = '-29.45,-51.98'

# upperleft = '37.8466,-122.2987'
# lowerright =  '37.8385,-122.2962'  

# zoom = 21   # be careful not to get too many images!
# #metersPerPx = 156543.03392 * Math.cos(latLng.lat() * Math.PI / 180) / Math.pow(2, zoom) (from google [https://gis.stackexchange.com/questions/7430/what-ratio-scales-do-google-maps-zoom-levels-correspond-to])

# ############################################

# ullat, ullon = map(float, upperleft.split(','))
# lrlat, lrlon = map(float, lowerright.split(','))

# # Set some important parameters
# scale = 1
# maxsize = 640

# # convert all these coordinates to pixels
# ulx, uly = latlontopixels(ullat, ullon, zoom)
# lrx, lry = latlontopixels(lrlat, lrlon, zoom)

# # calculate total pixel dimensions of final image
# dx, dy = lrx - ulx, uly - lry

# # calculate rows and columns
# cols, rows = int(ceil(dx/maxsize)), int(ceil(dy/maxsize))

# # calculate pixel dimensions of each small image
# bottom = 120
# largura = int(ceil(dx/cols))
# altura = int(ceil(dy/rows))
# alturaplus = altura + bottom


# final = Image.new("RGB", (int(dx), int(dy)))
# for x in range(cols):
#     for y in range(rows):
#         dxn = largura * (0.5 + x)
#         dyn = altura * (0.5 + y)
#         latn, lonn = pixelstolatlon(ulx + dxn, uly - dyn - bottom/2, zoom)
#         position = ','.join((str(latn), str(lonn)))
#         #print x, y, position
#         urlparams = urllib.parse.urlencode({'center': position,
#                                       'zoom': str(zoom),
#                                       'size': '%dx%d' % (largura, alturaplus),
#                                       'maptype': 'satellite',
#                                       'sensor': 'false',
#                                       'scale': scale,
#                                       'key':'AIzaSyBuYwhrUegFSvy2UaLDXd52CxiuXlMsLm4'})
#         url = 'http://maps.google.com/maps/api/staticmap?' + urlparams
#         f=urllib.request.urlopen(url)
#         #buffer = StringIO(f.read())
#         im=Image.open(io.BytesIO(f.read()))
#         final.paste(im, (int(x*largura), int(y*altura)))

# #final.show()

# opencvImage = np.array(final)
# cv2.imwrite('Map.png', opencvImage)
# cv2.namedWindow('test', cv2.WINDOW_NORMAL)
# cv2.imshow('test',opencvImage)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#Load the vehicle data
loadFileName = '/home/saptarshi/PythonCode/AdvanceLSTM/SplittedData/vehicle6.csv'
loadFile = open(loadFileName, 'r')
loadReader = csv.reader(loadFile)
loadDataset = []
for loadRow in loadReader:
        loadDataset.append([loadRow[6],loadRow[7]])

loadDataset.pop(0)
datasetArray = np.array(loadDataset, dtype=np.float)
xPosition = datasetArray[:,0]
yPosition = datasetArray[:,1]
cv2.namedWindow('test', cv2.WINDOW_NORMAL)

#Load the map
mapImage = cv2.imread('/home/saptarshi/PythonCode/AdvanceLSTM/Map.png')
w = mapImage.shape[1]
h = mapImage.shape[0]
xCentre = w/2
yCentre = h/2

midLon = -122.2962 - ((-122.2962 + 122.2987)/2)
midLat = 37.8385 + ((37.8466-37.8385)/2)
cornerLat = 37.8466
cornerLon = -122.2987
midPixelX, midPixelY = latlontopixels(midLat, midLon, 21)
cornerPixelX, cornerPixelY = latlontopixels(cornerLat, cornerLon, 21)


inProj = Proj(init='epsg:2227')  #, preserve_units = True)
outProj = Proj(init='epsg:4326')
for idx in range(0,len(xPosition)):
        x1,y1 = xPosition[idx],yPosition[idx]
        lon,lat = transform(inProj,outProj,x1,y1)
        pX,pY = latlontopixels(lat, lon, 21)
        dx = int(cornerPixelX - pX)*-1
        dy = int(cornerPixelY - pY)
        # xRegion = int(xCentre - dx )
        # yRegion = int(yCentre + dy)
        croppedMap = mapImage[dy-500:dy+500,dx-1000:dx+1000]
        croppedHeight = croppedMap.shape[0]
        croppedWidth = croppedMap.shape[1]
        center = (croppedHeight / 2, croppedWidth / 2)
        angle = 0
        scale = 1.0
        M = cv2.getRotationMatrix2D(center, angle, 2)
        #rotated90 = cv2.warpAffine(croppedMap, M, (h, w))
        rotated = imutils.rotate_bound(croppedMap, angle)
        cv2.rectangle(croppedMap,(int((croppedWidth/2)-40),int((croppedHeight/2)-40)),(int((croppedWidth/2)-20),int((croppedHeight/2)+20)),(0,255,0),3)
        cv2.imshow('test', croppedMap)
        cv2.waitKey(1)
