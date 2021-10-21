import os
import PIL
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import skimage
import math
from scipy.io import savemat
import csv
import shutil
from math import log, exp, tan, atan, pi, ceil, cos, sin
from pyproj import Proj, transform
from scipy import dot
import imutils


# Specify the test trajectory csv file
testTrajFilePath = '/home/saptarshi/PythonCode/AdvanceLSTM/SplittedData/Lankershim.csv'

# Set the different Occupancy Grid map and scene dimensions

# Create the visible window
cv2.namedWindow('test', cv2.WINDOW_NORMAL)


# Convert Lat lon to pixel
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



# Create the two dictionaries one based on FrameID and other based on VehicleID
def CreateVehicleAndFrameDict(loadFileName):

    print('Creating Vehicle and Frame based dictionary')

    loadFile = open(loadFileName, 'r')
    loadReader = csv.reader(loadFile)
    loadDataset = []
    for loadRow in loadReader:
        # if (loadRow[0] == '738' or loadRow[0] == '1755'): # remove two extreme car for better resolution
        #     continue
        loadDataset.append(loadRow[0:24])

    loadDataset.pop(0)
    sortedList = sorted(loadDataset, key=lambda x: (float(x[0]), float(x[1])))
    datasetArray = np.array(sortedList, dtype=np.float)

    #Create Dictionary for Mapper
    mapper = dict()

    # Create Dictionary with unique Frames
    uniquFrameIds = list(np.unique(datasetArray[:,1]))
    frameKeys = []
    for idx in range(0, len(uniquFrameIds)):
        frameKeys.append(str(uniquFrameIds[idx]))

    dictionaryByFrames = {key : list() for key in frameKeys}

    for jdx in range(0,len(datasetArray)):
        key = str(datasetArray[jdx,1])
        dictionaryByFrames[key].append(datasetArray[jdx])

    # Create Dictionary with unique Vehicles
    uniquVehicleIds = list(np.unique(datasetArray[:,0]))
    vehicleKeys = []
    for idx in range(0, len(uniquVehicleIds)):
        vehicleKeys.append(str(uniquVehicleIds[idx]))

    dictionaryByVehicles = {key : list() for key in vehicleKeys}

    for jdx in range(0,len(datasetArray)):
        key = str(datasetArray[jdx,0])
        if len(dictionaryByVehicles[key])==0:
            dictionaryByVehicles[key].append(datasetArray[jdx])
            continue
        lastFrame = dictionaryByVehicles[key][-1][1]
        lastTime = dictionaryByVehicles[key][-1][3]
        currentFrame = datasetArray[jdx][1]
        currentTime = datasetArray[jdx][3]
        if(abs(currentFrame-lastFrame)==1 and abs(currentTime-lastTime)==100):
            dictionaryByVehicles[key].append(datasetArray[jdx])
        else:
            if key in mapper:
                updatedKey = mapper[key]
                lastFrame = dictionaryByVehicles[updatedKey][-1][1]
                lastTime = dictionaryByVehicles[updatedKey][-1][3]
                currentFrame = datasetArray[jdx][1]
                currentTime = datasetArray[jdx][3]
                if(abs(currentFrame-lastFrame)==1 and abs(currentTime-lastTime)==100):                    
                    dictionaryByVehicles[updatedKey].append(datasetArray[jdx])
                else:
                    print('Wrong Assumption regarding the  presensce of one vehicle ID exists only twice...')
                    print('The problem occured for vehicle ID: ' + key + ' at frame: ' + str(currentFrame) + '...')
                    sys.exit()
            else:
                currentKeys = list(dictionaryByVehicles.keys())
                currentKeys.sort(key=float)
                newKey = str(float(currentKeys[-1]) + 1)
                mapper[key] = newKey
                dictionaryByVehicles[newKey] = list()
                dictionaryByVehicles[newKey].append(datasetArray[jdx])

    loadFile.close()

    return dictionaryByFrames,dictionaryByVehicles

# Plot all cars trajectory on the global GPS map
def PlotAllCars(inputFileName):

    #Load the map 
    mapImage = cv2.imread('/home/saptarshi/PythonCode/AdvanceLSTM/Maps/Lanekrshim.png')

    dictByFrames,dictByVehicles = CreateVehicleAndFrameDict(inputFileName)
    finalVehicleKeys = list(dictByVehicles.keys())
    finalVehicleKeys.sort(key=float)
    finalFrameKeys = list(dictByFrames.keys())
    finalFrameKeys.sort(key=float)

    intitalTime = 1118936700000.0    # Peachtree -> 1163030500 ----------- Lankershim -> 1118936700000.0 -> junc  ->  1118935680200.0

    # Create the projection from State plane to lat/lon
    inProj = Proj(init='epsg:2229', preserve_units = True)
    outProj = Proj(init='epsg:4326')

    cornerLat = 34.143
    cornerLon = -118.363
    cornerPixelX, cornerPixelY = latlontopixels(cornerLat, cornerLon, 21)

    targetSections = [2,3]
    targetIntersections = [2]

    straightCount = 0
    leftTurnCount = 0
    rightTurnCount = 0
    processedIds = []

    for currentFrame in finalFrameKeys:
        currentVehicleList = dictByFrames[str(currentFrame)]
        print('Processing Frame : ' + str(currentFrame))
        visImage = mapImage.copy()
        for eachCurrentVehicle in currentVehicleList:
            vehicleTimeStamp = eachCurrentVehicle[3]
            currentSection = eachCurrentVehicle[17]
            currentIntersection = eachCurrentVehicle[16]
            if ((vehicleTimeStamp == intitalTime) and ((currentSection in targetSections) or (currentIntersection in targetIntersections))):
                vehicleID = eachCurrentVehicle[0]
                globalX = eachCurrentVehicle[6]
                globalY = eachCurrentVehicle[7]
                localX = eachCurrentVehicle[4]
                movement = eachCurrentVehicle[19]
                destinationZone = eachCurrentVehicle[15]

                lon,lat = transform(inProj,outProj,globalX,globalY)
                pX,pY = latlontopixels(lat, lon, 21) 
                dx = int(cornerPixelX - pX )*-1 - 80
                dy = int(cornerPixelY - pY)

                color = (255,0,0)
                leftTurn = ((currentSection == 2) and (destinationZone == 211)) or ((currentSection == 3) and (destinationZone == 203)) or movement == 2
                rightTurn = ((currentSection == 2) and (destinationZone == 203)) or ((currentSection == 3) and (destinationZone == 211)) or movement == 3
                if (leftTurn):
                    color = (0,255,0)
                if (rightTurn):
                    color = (0,0,255)

                if vehicleID not in processedIds:
                    processedIds.append(vehicleID)
                    if(leftTurn):
                        leftTurnCount = leftTurnCount + 1
                    elif(rightTurn):
                        rightTurnCount = rightTurnCount + 1
                    else:
                        straightCount = straightCount + 1

                visImage = cv2.circle(visImage, (dx,dy), 12, color, -1)

        intitalTime = intitalTime + 100
        displayImage = visImage[5062:8724,400:1749]
        displayImage = cv2.rotate(displayImage,cv2.ROTATE_90_CLOCKWISE)
        fontScale = 2
        thickness = 8
        font = cv2.FONT_HERSHEY_SIMPLEX 
        blueTextColor = (255, 0, 0) 
        greenTextColor = (0, 255, 0) 
        redTextColor = (0, 0, 255) 
        displayImage = cv2.putText(displayImage, 'Straight :', (2000,180), font, fontScale, blueTextColor, thickness, cv2.LINE_AA) 
        displayImage = cv2.putText(displayImage, 'Left Turn :', (2500,180), font, fontScale, greenTextColor, thickness, cv2.LINE_AA) 
        displayImage = cv2.putText(displayImage, 'Right Turn :', (3000,180), font, fontScale, redTextColor, thickness, cv2.LINE_AA) 

        straightCountStr = str(straightCount)
        leftTurnCountStr = str(leftTurnCount)
        rightTurnCountStr = str(rightTurnCount)

        displayImage = cv2.putText(displayImage, straightCountStr, (2325,180), font, fontScale, blueTextColor, thickness, cv2.LINE_AA) 
        displayImage = cv2.putText(displayImage, leftTurnCountStr, (2850,180), font, fontScale, greenTextColor, thickness, cv2.LINE_AA) 
        displayImage = cv2.putText(displayImage, rightTurnCountStr, (3400,180), font, fontScale, redTextColor, thickness, cv2.LINE_AA) 

        if (float(currentFrame) > 400 and float(currentFrame) < 5000):
            cv2.imwrite('/home/saptarshi/PythonCode/AdvanceLSTM/JunctionVisual/' + str(currentFrame) + '.png', displayImage)
        cv2.imshow('test', displayImage)
        cv2.waitKey(1)

if __name__ == '__main__':

    PlotAllCars(testTrajFilePath)

    print('All the cars are plotted in the scene.')

    sys.exit()







