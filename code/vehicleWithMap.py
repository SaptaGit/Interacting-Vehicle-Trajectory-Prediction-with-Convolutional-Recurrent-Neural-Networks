import sys
import csv
import numpy as np
from PIL import Image
import os
from math import log, exp, tan, atan, pi, ceil, cos, sin
import math
import csv
from pyproj import Proj, transform
import cv2
from scipy import dot

feetPerPx = 156543.03392 * math.cos(37.84255 * math.pi / 180) / math.pow(2, 21) * 3.28084
pxPerFeet = 1/feetPerPx
widthPixel = int(pxPerFeet*2)
heightPixel = int(pxPerFeet*4)


def DrawAngledRec(x0, y0, width, height, angle, img, color):

    _angle = angle * math.pi / 180.0
    b = math.cos(_angle) * 0.5
    a = math.sin(_angle) * 0.5
    pt0 = (int(x0 - a * height - b * width),
           int(y0 + b * height - a * width))
    pt1 = (int(x0 + a * height - b * width),
           int(y0 - b * height - a * width))
    pt2 = (int(2 * x0 - pt0[0]), int(2 * y0 - pt0[1]))
    pt3 = (int(2 * x0 - pt1[0]), int(2 * y0 - pt1[1]))

    cv2.line(img, pt0, pt1, color, 3)
    cv2.line(img, pt1, pt2, color, 3)
    cv2.line(img, pt2, pt3, color, 3)
    cv2.line(img, pt3, pt0, color, 3)

folderCount = 0

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


def LoadDataSetAndSplit():
    print('loading CSV')
    dataFile = open('/home/saptarshi/PythonCode/AdvanceLSTM/Next_Generation_Simulation__NGSIM__Vehicle_Trajectories_and_Supporting_Data.csv', 'r')
    reader = csv.reader(dataFile)
    next(reader, None) #Skip header row
    dataSet = []
    #cnt = 0
    for row in reader:
        if (row[24] == 'i-80'):
            dataSet.append(row)
        
    print('Loaded in array') 

    sortedList = sorted(dataSet, key=lambda x: (int(x[1]), int(x[0])))
    dataSetArraySorted = np.array(sortedList)

    print('sorted in array')

    rowLength = len(dataSet)
    numbeOfFiles = 20
    fileSize = int(rowLength/numbeOfFiles)
    count = 0

    for idx in range(0,numbeOfFiles):
        requiredItems = dataSetArraySorted[count:count+fileSize,0:14]
        fileItems = np.array(requiredItems, dtype=np.float) 
        print(fileItems.shape)
        fileName = 'SplittedData/' + str(count).zfill(6) + '.csv'
        with open(fileName, 'w') as currentFile:
            np.savetxt(currentFile, fileItems, fmt = '%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f')
        count = count + fileSize

    dataFile.close()

def createRelativePositionList(loadFileName):

    print('Creating Relative Position List')

    loadFile = open(loadFileName, 'r')
    loadReader = csv.reader(loadFile)
    loadDataset = []
    for loadRow in loadReader:
        loadDataset.append(loadRow)

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

    #Each list in this list is [vehicleId, frameID, ]
    relativePositionList = []
    finalVehicleKeys = list(dictionaryByVehicles.keys())
    finalVehicleKeys.sort(key=float)

    for keyVal in finalVehicleKeys:
        vehicleData = dictionaryByVehicles[keyVal]
        vehicleSpecificList = []
        for vehicleVal in vehicleData:
            VehicleID = vehicleVal[0]
            FrameID = vehicleVal[1]
            VehicleX = vehicleVal[4]
            VehicleY = vehicleVal[5]
            VehicleGX = vehicleVal[6]
            VehicleGY = vehicleVal[7]
            VehicleLength = vehicleVal[8]
            VehicleWidth = vehicleVal[9]
            VehicleTime = vehicleVal[3]
            frameData = dictionaryByFrames[str(FrameID)]
            frameSpecificList = []
            frameSpecificList.append([VehicleID, FrameID])
            frameSpecificList.append([VehicleX, VehicleY, VehicleLength, VehicleWidth, VehicleTime, VehicleGX, VehicleGY])
            for frameVal in frameData:
                frameSpecificVehicleID = frameVal[0]
                currentTime = frameVal[3]
                if (frameSpecificVehicleID != VehicleID) and (VehicleTime==currentTime):
                    currentX = frameVal[4]
                    currentY = frameVal[5]
                    currentGX = frameVal[6]
                    currentGY = frameVal[7]
                    currentLength = frameVal[8]
                    currentWidth = frameVal[9]
                    currentTime = frameVal[3]
                    frameSpecificList.append([currentX, currentY, currentLength, currentWidth, currentTime, currentGX, currentGY])
            vehicleSpecificList.append(frameSpecificList)
        relativePositionList.append(vehicleSpecificList)
    
    loadFile.close()

    print('Relative Position List created for one file')

    return relativePositionList

def generateOccupancyGrid(vehiclePositions):

    cv2.namedWindow('test', cv2.WINDOW_NORMAL)
    print('Generating the Occupancy Grid Maps')
    #Load the map
    mapImage = cv2.imread('/home/saptarshi/PythonCode/AdvanceLSTM/Maps/MapCarRemoval.png')
    cornerLat = 37.8466
    cornerLon = -122.2987
    cornerPixelX, cornerPixelY = latlontopixels(cornerLat, cornerLon, 21)
    inProj = Proj(init='epsg:2227', preserve_units = True)
    outProj = Proj(init='epsg:4326')

    occupancyMapWidth = 100
    occupancyMapHeight = 600
    OccupancyImageWidth = 256
    OccupancyImageHeight = 1024
    imageOriginX = OccupancyImageWidth/2
    imageOriginY = OccupancyImageHeight/2
    lonResolution = OccupancyImageHeight/occupancyMapHeight
    latResolution = OccupancyImageWidth/occupancyMapWidth
    imageCount = 0
    sampleSize = 60
    inputSize = 30
    outputSize = 60
    imageFolder = '../OccupancyMaps/'
    imageFileType = '.jpeg'
    for vehicle in vehiclePositions:
        for idx, _ in enumerate(vehicle):
            x1,y1 = vehicle[idx][1][5], vehicle[idx][1][6]
            lon,lat = transform(inProj,outProj,x1,y1)
            pX,pY = latlontopixels(lat, lon, 21)
            dx = int(cornerPixelX - pX)*-1
            dy = int(cornerPixelY - pY)
            length = int(vehicle[idx][1][2]*pxPerFeet)
            width = int(vehicle[idx][1][3]*pxPerFeet)
            showImage = np.copy(mapImage)
            imageCentreX, imageCentreY = dx, dy
            DrawAngledRec(dx-width,dy,width,length,math.radians(-410),showImage, (0,0,255))
            for kdx in range(2,len(vehicle[idx])):
                x1,y1 = vehicle[idx][kdx][5], vehicle[idx][kdx][6]
                lon,lat = transform(inProj,outProj,x1,y1)
                pX,pY = latlontopixels(lat, lon, 21)
                dx = int(cornerPixelX - pX)*-1
                dy = int(cornerPixelY - pY)
                length = int(vehicle[idx][1][2]*pxPerFeet)
                width = int(vehicle[idx][1][3]*pxPerFeet)
                DrawAngledRec(dx-width,dy,width,length,math.radians(-410),showImage, (0,255,0))
            
            croppedImage = showImage[dy-1000:dy+1000,dx-500:dx+500]
            rotated = np.rot90(croppedImage,3)
            cv2.imshow('test', rotated)
            cv2.waitKey(1)



if __name__ == '__main__':

    # Load the dataset and split into multiple files for future use
    #LoadDataSetAndSplit()


    # dataFileList = sorted(os.listdir('SplittedData'))
    # sortedDataFiles = sorted(dataFileList, key=lambda a: int(a.split(".")[0]) )
    # for dataFile in sortedDataFiles:
    #     generateOccupancyGrid(createRelativePositionList('SplittedData/' + dataFile))

    # print('All the occupancy grid map generated!!!!!!')

    # print('check sys exit')

    # if(os.path.isdir('OccupancyMaps')):
    #     print('Occupancy folder exist. What to do with the current one??')
    #     sys.exit()
    # else:
    #     os.mkdir('OccupancyMaps')

    #testList=createRelativePositionList('/home/saptarshi/PythonCode/AdvanceLSTM/SplittedData/maptest.csv')
    testList=createRelativePositionList('/home/saptarshi/PythonCode/AdvanceLSTM/SplittedData/maptest.csv')
    generateOccupancyGrid(testList)

