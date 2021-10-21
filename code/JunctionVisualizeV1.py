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
testTrajFilePath = '/home/saptarshi/PythonCode/AdvanceLSTM/SplittedData/junc.csv'

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

    #Map physical height and width calculated using lan lot based distance calculator need double check
    physicalHeight = 1600  #    1660    #3645
    physicalWidth = 200

    #Load the map and extract the height and width to solve the resoultion
    mapImage = cv2.imread('/home/saptarshi/PythonCode/AdvanceLSTM/Maps/trajs.png')
    mapHeight = mapImage.shape[0]
    mapWidth = mapImage.shape[1]

    mapImage = imutils.rotate(mapImage, -22)

    #cv2.imwrite('/home/saptarshi/PythonCode/AdvanceLSTM/Maps/LanekrshimRotate.png', mapImage)

    cv2.imshow("test", mapImage)
    cv2.waitKey(1)

    # Calculate the height and width resolution pixel/feet
    heightRes = mapHeight/physicalHeight
    widthRes = mapWidth/physicalWidth


    dictByFrames,dictByVehicles = CreateVehicleAndFrameDict(inputFileName)
    finalVehicleKeys = list(dictByVehicles.keys())
    finalVehicleKeys.sort(key=float)
    finalFrameKeys = list(dictByFrames.keys())
    finalFrameKeys.sort(key=float)

    # singleVehicle = dictByVehicles['63.0']
    # singlePoseList = []
    # for eachPoint in singleVehicle:
    #     singlePoseList.append((eachPoint[4],eachPoint[5]))

    # singlePostArray = np.array(singlePoseList)

    # plt.scatter(singlePostArray[:,1],singlePostArray[:,0])
    # plt.show()




    intitalTime = 1118936700000.0    # Peachtree -> 1163030500 ----------- Lankershim -> 1118936700000.0 -> junc  ->  1118935680200.0
    minX = 9999
    maxX = 0

    img = np.zeros([1600,830,3],dtype=np.uint8)
    img.fill(255)

    inProj = Proj(init='epsg:2229', preserve_units = True)
    outProj = Proj(init='epsg:4326')

    for currentFrame in finalFrameKeys:
        currentVehicleList = dictByFrames[str(currentFrame)]
        print('Processing Frame : ' + str(currentFrame))
        for eachCurrentVehicle in currentVehicleList:
            vehicleTimeStamp = eachCurrentVehicle[3]
            if vehicleTimeStamp == intitalTime:
                # draw each vehicle in blank map
                # if (eachCurrentVehicle[4] > maxX):
                #     maxX = eachCurrentVehicle[4]
                # if (eachCurrentVehicle[4] < minX):
                #     minX = eachCurrentVehicle[4]
                # poseX = int((mapWidth/2) - eachCurrentVehicle[4]*widthRes)        #int(((eachCurrentVehicle[4]+120)/240)*500) # re scalling for propoer seperation in lanes.
                # poseY = int(eachCurrentVehicle[5]*heightRes)
                # poseX = int((mapWidth/2) + eachCurrentVehicle[4]*widthRes)
                # poseY = int((physicalHeight-eachCurrentVehicle[5])*heightRes)
                poseX = int((830/2) + eachCurrentVehicle[4])
                poseY = int((1600-eachCurrentVehicle[5]))

                lon,lat = transform(inProj,outProj,6451969.763,1872795.261)   #6451967.185   1872797.378

                # pX,pY = latlontopixels(lat, lon, 21)
                # if (poseX > 450 or poseX < 200):
                #     img = cv2.putText(img, str(eachCurrentVehicle[0]), (poseY, poseX), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
                #     cv2.waitKey(0)
                img = cv2.circle(img, (poseX,poseY), 2, (0, 0, 255))
                #img = cv2.putText(img, str(poseX), (poseY, poseX), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
        intitalTime = intitalTime + 100
        cv2.imshow('test', img)
        cv2.waitKey(10)
    cv2.imwrite('/home/saptarshi/PythonCode/AdvanceLSTM/Maps/trajs.png', img)
    
    # print('Min X = ' + str(minX))
    # print('Max X = ' + str(maxX))


                



if __name__ == '__main__':

    PlotAllCars(testTrajFilePath)

    print('All the cars are plotted in the scene.')

    sys.exit()







