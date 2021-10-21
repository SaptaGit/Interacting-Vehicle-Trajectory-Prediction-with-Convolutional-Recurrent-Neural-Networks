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


# Specify the test trajectory csv file
testTrajFilePath = '/home/saptarshi/PythonCode/AdvanceLSTM/SplittedData/junc.csv'
# This one is to pass some specific vehicle ids as list for situations like if lane change vehicles or turn vehicles.
#LaneChangeVehicleList = [11.0,21.0,7.0,5.0,21.0,87.0,44.0,54.0,50.0,41.0,31.0,115.0,32.0,45.0,121.0,144.0]
LaneChangeVehicleList = [142.0,159.0,169.0]
# Set the different Occupancy Grid map and scene dimensions
occupancyMapWidth = 100
occupancyMapHeight = 600
OccupancyImageWidth = 128
OccupancyImageHeight = 512
imageOriginX = OccupancyImageWidth/2
imageOriginY = OccupancyImageHeight/2
lonResolution = OccupancyImageHeight/occupancyMapHeight
latResolution = OccupancyImageWidth/occupancyMapWidth
sampleSize = 60
inputSize = 30
outputSize = 60
channel = 1
temporal = 30
targetCarFolder = 'target'
otherCarFolder = 'other'
imageFileType = '.png'
globalImageWidth = 400
globalImageHeight = 3600
globalFeetWidth = 100
globalFeetHeight = 1800
globalWidthResolution = globalImageWidth/globalFeetWidth
globalHeightResolution = globalImageHeight/globalFeetHeight
feetToMeter = 0.3048
globalErrorCount = 0

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

# Create the two dictionaries one based on FrameID and other based on VehicleID
def CreateVehicleAndFrameDict(loadFileName):

    print('Creating Vehicle and Frame based dictionary')

    loadFile = open(loadFileName, 'r')
    loadReader = csv.reader(loadFile)
    loadDataset = []
    for loadRow in loadReader:
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

    #Load the map and other map related parameters
    mapImage = cv2.imread('/home/saptarshi/PythonCode/AdvanceLSTM/Maps/MapCarRemoval.png')
    cornerLat = 37.8466
    cornerLon = -122.2987
    cornerPixelX, cornerPixelY = latlontopixels(cornerLat, cornerLon, 21)
    inProj = Proj(init='epsg:2227', preserve_units = True)
    outProj = Proj(init='epsg:4326')
    LaneChangeCount = 0
    FollowRoadCount = 0

    dictByFrames,dictByVehicles = CreateVehicleAndFrameDict(inputFileName)
    finalVehicleKeys = list(dictByVehicles.keys())
    finalVehicleKeys.sort(key=float)
    finalFrameKeys = list(dictByFrames.keys())
    finalFrameKeys.sort(key=float)

    relativePositionList = []

    for keyVal in finalVehicleKeys:
        vehicleData = dictByVehicles[keyVal]
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
            LaneNumber = vehicleVal[13]
            frameData = dictByFrames[str(FrameID)]
            frameSpecificList = []
            frameSpecificList.append([VehicleID, FrameID])
            frameSpecificList.append([VehicleX, VehicleY, VehicleLength, VehicleWidth, VehicleTime, VehicleGX, VehicleGY, LaneNumber])
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
                    otherLaneNumber = frameVal[13]
                    frameSpecificList.append([currentX, currentY, currentLength, currentWidth, currentTime, frameSpecificVehicleID, currentGX, currentGY, otherLaneNumber])
            vehicleSpecificList.append(frameSpecificList)
        relativePositionList.append(vehicleSpecificList)

    # Create the visible window
    cv2.namedWindow('test', cv2.WINDOW_NORMAL)
    

    for vehicle in relativePositionList:
        currentVehicleID = vehicle[0][0][0]
        # if(currentVehicleID not in LaneChangeVehicleList):
        #     continue
        trackerDict = dict()
        print('Plotting for vehicle ID: ' + str(vehicle[0][0][0]))
        for idx, _ in enumerate(vehicle):
            if (idx >= len(vehicle)):
                break

            inputSampleFrames = vehicle[idx]
            primaryCarId = inputSampleFrames[0][0]
            currentTargetX, currentTargetY, length, width, localTime, VehicleGX, VehicleGY, targetLaneNumber = inputSampleFrames[1]
            carsInCurrentFrame = []
            carsInCurrentFrame.append(primaryCarId)
            if primaryCarId in trackerDict:
                trackerDict[primaryCarId].append([currentTargetX, currentTargetY, length, width, localTime, VehicleGX, VehicleGY, targetLaneNumber])
            else:
                trackerDict[primaryCarId] = [[currentTargetX, currentTargetY, length, width, localTime, VehicleGX, VehicleGY, targetLaneNumber]]

            frameLength = len(inputSampleFrames)
            for jdx in range(2,frameLength):
                if (localTime == inputSampleFrames[jdx][4]):
                    otherCarId = inputSampleFrames[jdx][5]
                    carsInCurrentFrame.append(otherCarId)
                    if otherCarId in trackerDict:
                        trackerDict[otherCarId].append(inputSampleFrames[jdx])
                    else:
                        trackerDict[otherCarId] = [inputSampleFrames[jdx]]

            existingCars = list(trackerDict.keys())
            deathCars = list(set(existingCars) - set(carsInCurrentFrame))
            #delete the cars which are death cars from the trakcer
            for death in deathCars:
                del trackerDict[death]

            if ( len(trackerDict[primaryCarId]) == 30):
                targetCarPosesList = list(np.array(trackerDict[primaryCarId])[:,5:7])
                targetLane = np.array(trackerDict[primaryCarId])[-1,7]
                laneCount = len(np.unique(np.array(trackerDict[primaryCarId])[:,7]))
                if laneCount == 1:
                    laneChangeStatus = 'No'
                    FollowRoadCount = FollowRoadCount + 1
                else:
                    laneChangeStatus = 'Yes'
                    LaneChangeCount = LaneChangeCount + 1
                targetPosesTransformedList = []
                showImage = np.copy(mapImage)
            
                for ldx,sampleTargetFrame in enumerate(targetCarPosesList):
                    x1,y1 = targetCarPosesList[ldx]
                    lon,lat = transform(inProj,outProj,x1,y1)
                    pX,pY = latlontopixels(lat, lon, 21)
                    dx = int(cornerPixelX - pX)*-1
                    dy = int(cornerPixelY - pY)
                    targetPosesTransformedList.append([dx,dy])
                imageCentreX,imageCentreY = dx,dy
                pts = np.array(targetPosesTransformedList, np.int32)
                pts = pts.reshape((-1,1,2))
                cv2.polylines(showImage,[pts],False,(0,0,255),3)
                #cv2.putText(showImage, str(targetLane), (int(dx),int(dy)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),1)
                cv2.putText(showImage, str(primaryCarId), (int(dx),int(dy)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),1)

                for otherTrackedEligibleCarKey in trackerDict:
                    if otherTrackedEligibleCarKey != primaryCarId and len(trackerDict[otherTrackedEligibleCarKey]) == 30:
                        otherCarPosesList = list(np.array(trackerDict[otherTrackedEligibleCarKey])[:,6:8])
                        otherLane = np.array(trackerDict[otherTrackedEligibleCarKey])[-1,8]
                        laneCount = len(np.unique(np.array(trackerDict[otherTrackedEligibleCarKey])[:,8]))
                        if laneCount == 1:
                            laneChangeStatus = 'No'
                            FollowRoadCount = FollowRoadCount + 1
                        else:
                            laneChangeStatus = 'Yes'
                            LaneChangeCount = LaneChangeCount + 1
                        otherCarPosesListPosesTransformedList = []

                        for kdx,sampleOtherFrame in enumerate(otherCarPosesList):
                            x1,y1 = otherCarPosesList[kdx]
                            lon,lat = transform(inProj,outProj,x1,y1)
                            pX,pY = latlontopixels(lat, lon, 21)
                            dx = int(cornerPixelX - pX)*-1
                            dy = int(cornerPixelY - pY)
                            otherCarPosesListPosesTransformedList.append([dx,dy])
                        
                        pts = np.array(otherCarPosesListPosesTransformedList, np.int32)
                        pts = pts.reshape((-1,1,2))
                        cv2.polylines(showImage,[pts],False,(0,0,255),3)
                        #cv2.putText(showImage, str(otherLane), (int(dx),int(dy)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),1)
                        cv2.putText(showImage, str(otherTrackedEligibleCarKey), (int(dx),int(dy)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),1)
                        # Remove the old car pose from the tracker dict for other cars
                        trackerDict[otherTrackedEligibleCarKey].pop(0)

                # Remove the old car pose from the tracker dict for target cars
                trackerDict[primaryCarId].pop(0)

                print('Follow Road Count : ' + str(FollowRoadCount) + ' Lane Change Count : ' + str(LaneChangeCount))

                # Display the plotted trajectories 
                croppedImage = showImage[imageCentreY-1000:imageCentreY+1000,imageCentreX-500:imageCentreX+500]
                rotated = np.rot90(croppedImage,3)
                cv2.imshow('test', rotated)
                cv2.waitKey(1)

                



if __name__ == '__main__':

    PlotAllCars(testTrajFilePath)

    print('All the cars are plotted in the scene.')

    sys.exit()







