import sys
import csv
import numpy as np
from PIL import Image
import os
from math import log, exp, tan, atan, pi, ceil, cos, sin
from pyproj import Proj, transform
from scipy import dot
import cv2
import imutils
import math
import random

# This is to count how many samples created and also create new folder name each 
# time a new sample is created
folderCount = 0
# This is the absolute path for where the generated occupancy map will be saved
imageFolder = '/home/saptarshi/PythonCode/AdvanceLSTM/foldertest/'
# This is the csv file absolute path for which the training OGM maps will be generated
trainingCsvFile = '/home/saptarshi/PythonCode/AdvanceLSTM/SplittedData/vehicle7.csv'

# Internal used variables:
occupancyMapWidth = 100
occupancyMapHeight = 600
OccupancyImageWidth = 128
OccupancyImageHeight = 1024
imageOriginX = OccupancyImageWidth/2
imageOriginY = OccupancyImageHeight/2
lonResolution = OccupancyImageHeight/occupancyMapHeight
latResolution = OccupancyImageWidth/occupancyMapWidth
sampleSize = 60
inputSize = 30
outputSize = 60
imageFileType = '.png'

# Background Map related dimensions
feetPerPx = 156543.03392 * math.cos(37.84255 * math.pi / 180) / math.pow(2, 21) * 3.28084
pxPerFeet = 1/feetPerPx
OgmLatPixels = int(occupancyMapWidth*pxPerFeet)
OgmLonPixels = int(occupancyMapHeight*pxPerFeet)
OgmLatPixelsMargin = 100
OgmLonPixelsMargin = 100
lonPixelCount = int((OgmLonPixels/2)+OgmLonPixelsMargin)
latPixelCount = int((OgmLatPixels/2)+OgmLatPixelsMargin)
mapRotationAngle = -8.5

# This one is to pass some specific vehicle ids as list for situations like if lane change vehicles or turn vehicles.
LaneChangeVehicleList = [11.0,21.0,7.0,5.0,21.0,87.0,44.0,54.0,50.0,41.0,31.0,115.0,32.0,45.0,121.0,60.0,144.0]
normalVehicleCount = 25
for i in range(normalVehicleCount):
    LaneChangeVehicleList.append(random.randint(0,150))


# Convert latitude longitude to pixel units
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



# Create a list of cars. Where each car is also a list which holds the position 
# of that car for each frame and also the surrounding cars position as well. 
# This list will be used later to create the occuoancy maps for each specific vehicle.
def createRelativePositionList(loadFileName):

    print('Creating Relative Position List!!!')

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
            TargetGlobalX = vehicleVal[6]
            TargetGlobalY = vehicleVal[7]
            VehicleLength = vehicleVal[8]
            VehicleWidth = vehicleVal[9]
            VehicleTime = vehicleVal[3]
            LaneNumber = vehicleVal[13]
            frameData = dictionaryByFrames[str(FrameID)]
            frameSpecificList = []
            frameSpecificList.append([VehicleID, FrameID])
            frameSpecificList.append([VehicleX, VehicleY, VehicleLength, VehicleWidth, VehicleTime, TargetGlobalX, TargetGlobalY, LaneNumber])
            for frameVal in frameData:
                frameSpecificVehicleID = frameVal[0]
                currentTime = frameVal[3]
                if (frameSpecificVehicleID != VehicleID) and (VehicleTime==currentTime):
                    currentX = frameVal[4]
                    currentY = frameVal[5]
                    otherGlobalX = frameVal[6]
                    otherGlobalY = frameVal[7]
                    currentLength = frameVal[8]
                    currentWidth = frameVal[9]
                    currentTime = frameVal[3]
                    otherLaneNumber = frameVal[13]
                    frameSpecificList.append([currentX, currentY, currentLength, currentWidth, currentTime, otherGlobalX, otherGlobalY, otherLaneNumber])
            vehicleSpecificList.append(frameSpecificList)
        relativePositionList.append(vehicleSpecificList)
    
    loadFile.close()

    print('Relative Position List created!!!')

    return relativePositionList


# This one receives the previously explained list and creates the occupancy map samples.
#  Each sample is one folder under parent folder "imageFolder". Under each sample folder 
# there are 50 image frames which are the occupancy grid maps w.r.t one specific vehicle 
# for the last 50 frames and one "output·txt" file which holds the position of that specific
#  target car for next 50 frames w.r.t to last OGM frame in image co-ordinate system. 

def generateOccupancyGrid(vehiclePositions):

    print('Generating the Occupancy Grid Maps!!!')

    #Load the map and other map related parameters
    mapImage = cv2.imread('/home/saptarshi/PythonCode/AdvanceLSTM/Maps/MapCarRemoval.png')
    cornerLat = 37.8466
    cornerLon = -122.2987
    cornerPixelX, cornerPixelY = latlontopixels(cornerLat, cornerLon, 21)
    inProj = Proj(init='epsg:2227', preserve_units = True)
    outProj = Proj(init='epsg:4326')

    imageCount = 0

    for vehicle in vehiclePositions:
        currentVehicleID = vehicle[0][0][0]
        print('Plotting for vehicle ID: ' + str(vehicle[0][0][0]))
        if(currentVehicleID not in LaneChangeVehicleList):
           continue
            
        for idx, _ in enumerate(vehicle):
            if (idx+outputSize > len(vehicle)):
                break
            global folderCount
            folderPath = imageFolder + str(folderCount)
            os.mkdir(folderPath)
            print('Processing occupancy sample:' + str(folderCount) + ' for vehicleID ' + str(currentVehicleID))
            folderCount = folderCount + 1
            inputSampleFrames = vehicle[idx:idx+inputSize]
            localOriginX = inputSampleFrames[-1][1][0]
            localOriginY = inputSampleFrames[-1][1][1]
            mapOriginX = inputSampleFrames[-1][1][5]
            mapOriginY = inputSampleFrames[-1][1][6]
            mapCentreLon,mapCentreLat = transform(inProj,outProj,mapOriginX,mapOriginY)
            mapPixelX,mapPixelY = latlontopixels(mapCentreLat, mapCentreLon, 21) 
            dx = int(cornerPixelX - mapPixelX)*-1
            dy = int(cornerPixelY - mapPixelY)
            #currentFrame =  mapImage[dy-512:dy+512,dx-128:dx+128]
            #cv2.imshow('test',sampleMapImage)
            #cv2.waitKey(0)
            for sampleFrame in inputSampleFrames:
                # Process the Background map OgmLatPixels
                #rawMap =  np.copy(mapImage[dy-lonPixelCount:dy+lonPixelCount,dx-latPixelCount+50:dx+latPixelCount+50])
                rawMap =  np.copy(mapImage[dy-lonPixelCount:dy+lonPixelCount,dx-latPixelCount:dx+latPixelCount])
                #rawMap =  np.copy(mapImage[dy-1600:dy+1600,dx-350:dx+350])
                rotatedMap = imutils.rotate(rawMap,mapRotationAngle)
                #rotatedCropped = rotatedMap[100:3100,100:600]
                rotatedCropped = rotatedMap[int(OgmLonPixelsMargin):int(lonPixelCount*2-OgmLonPixelsMargin),int(OgmLatPixelsMargin):int(latPixelCount*2-OgmLatPixelsMargin)]
                currentFrame = cv2.resize(rotatedCropped,(OccupancyImageWidth,OccupancyImageHeight))
                # cv2.imshow('test',currentFrame)
                # cv2.waitKey(0)

                #currentFrame = np.zeros((OccupancyImageHeight,OccupancyImageWidth), dtype=np.uint8)
                #currentFrame.fill(255)
                currentTargetX, currentTargetY, length, width, localTime, VehicleGX, VehicleGY, TargetLane = sampleFrame[1]
                #relativeTargetX = imageOriginX - int((currentTargetX - localOriginX)*latResolution)
                relativeTargetX = imageOriginX + int((currentTargetX - localOriginX)*latResolution)
                relativeTargetY = imageOriginY - int((currentTargetY - localOriginY)*lonResolution)
                pixelLength = int(length*lonResolution)
                pixelWidth = int((width*latResolution)/2)
                currentFrame[int(relativeTargetY):int(relativeTargetY+pixelLength),int(relativeTargetX-pixelWidth):int(relativeTargetX+pixelWidth)] = 0
                cv2.putText(currentFrame, str(TargetLane), (int(relativeTargetX),int(relativeTargetY)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0),1)
                frameLength = len(sampleFrame)
                for jdx in range(2,frameLength):
                    if (localTime == sampleFrame[jdx][4]):
                        #absoluteXPixel = imageOriginX - int((sampleFrame[jdx][0]-localOriginX)*latResolution)
                        absoluteXPixel = imageOriginX + int((sampleFrame[jdx][0]-localOriginX)*latResolution)
                        absoluteYPixel = imageOriginY - int((sampleFrame[jdx][1]-localOriginY)*lonResolution)
                        pixelLength = int(sampleFrame[jdx][2]*lonResolution)
                        pixelWidth = int((sampleFrame[jdx][3]*latResolution)/2)
                        currentFrame[int(absoluteYPixel):int(absoluteYPixel+pixelLength),int(absoluteXPixel-pixelWidth):int(absoluteXPixel+pixelWidth)] = 128
                        otherLane = sampleFrame[jdx][7]
                        cv2.putText(currentFrame, str(otherLane), (int(absoluteXPixel),int(absoluteYPixel)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0),1)
                im = Image.fromarray(currentFrame)
                imageCount = imageCount+1
                imFileName = folderPath + '/' + str(imageCount) + imageFileType
                im.save(imFileName)
            
            outputSampleFrames = vehicle[idx+inputSize+1:idx+outputSize+1]
            outFileName = folderPath + '/' + 'output.txt'
            f = open(outFileName, 'w')
            for outputSample in outputSampleFrames:
                currentTargetX, currentTargetY, length, width, localTime, VehicleGX, VehicleGY, TargetLane = outputSample[1]
                relativeOutputTargetX = imageOriginX - int((currentTargetX - localOriginX)*latResolution)
                relativeOutputTargetY = imageOriginY - int((currentTargetY - localOriginY)*lonResolution)
                outputString = str(relativeOutputTargetX) + ',' + str(relativeOutputTargetY) + '\n'
                f.writelines(outputString)
            f.close()
            
    print('All the occupancy grid map generated!!!!!!')


if __name__ == '__main__':


    if(os.path.isdir(imageFolder)):
        print(imageFolder + ' folder exists. What to do with the current one??')
        sys.exit()
    else:
        os.mkdir(imageFolder)

    testList=createRelativePositionList(trainingCsvFile)
    generateOccupancyGrid(testList)

