from PIL import Image
import urllib
import urllib.request
from io import StringIO
import io
from math import log, exp, tan, atan, pi, ceil
import numpy as np
import cv2


api_key = 'AIzaSyBuYwhrUegFSvy2UaLDXd52CxiuXlMsLm4'

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

############################################

# a neighbourhood in Lajeado, Brazil:

upperleft =  '34.143,-118.363'  
lowerright = '34.137,-118.360'

zoom = 21   # be careful not to get too many images!

############################################

ullat, ullon = map(float, upperleft.split(','))
lrlat, lrlon = map(float, lowerright.split(','))

# Set some important parameters
scale = 1
maxsize = 640

# convert all these coordinates to pixels
ulx, uly = latlontopixels(ullat, ullon, zoom)
lrx, lry = latlontopixels(lrlat, lrlon, zoom)

# calculate total pixel dimensions of final image
dx, dy = lrx - ulx, uly - lry

# calculate rows and columns
cols, rows = int(ceil(dx/maxsize)), int(ceil(dy/maxsize))

# calculate pixel dimensions of each small image
bottom = 120
largura = int(ceil(dx/cols))
altura = int(ceil(dy/rows))
alturaplus = altura + bottom


final = Image.new("RGB", (int(dx), int(dy)))
for x in range(cols):
    for y in range(rows):
        dxn = largura * (0.5 + x)
        dyn = altura * (0.5 + y)
        latn, lonn = pixelstolatlon(ulx + dxn, uly - dyn - bottom/2, zoom)
        position = ','.join((str(latn), str(lonn)))
        #print x, y, position
        urlparams = urllib.parse.urlencode({'center': position,
                                      'zoom': str(zoom),
                                      'size': '%dx%d' % (largura, alturaplus),
                                      'maptype': 'satellite',
                                      'sensor': 'false',
                                      'key' : api_key,
                                      'scale': scale})
        url = 'http://maps.google.com/maps/api/staticmap?' + urlparams
        f=urllib.request.urlopen(url).read()
        im=Image.open(io.BytesIO(f))
        final.paste(im, (int(x*largura), int(y*altura)))
openCvImage = np.array(final)

cv2.imwrite('/home/saptarshi/PythonCode/AdvanceLSTM/Maps/Lanekrshim.png', openCvImage)

cv2.namedWindow('map',cv2.WINDOW_NORMAL)
cv2.imshow('map', openCvImage)
cv2.waitKey(0)