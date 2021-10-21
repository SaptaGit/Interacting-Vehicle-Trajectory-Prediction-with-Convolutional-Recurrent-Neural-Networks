import sys
import csv
import numpy as np
from PIL import Image
import os
import cv2

# This is to count how many samples created and also create new folder name each 
# time a new sample is created
folderCount = 0
# This is the absolute path for where the generated occupancy map will be saved
#imageFolder = '/home/saptarshi/PythonCode/AdvanceLSTM/foldertest/'
# This is the csv file absolute path for which the training OGM maps will be generated
trainingCsvFile = '/home/saptarshi/PythonCode/AdvanceLSTM/SplittedData/i80-0400-0415.csv'

# Internal used variables:
occupancyMapWidth = 100
occupancyMapHeight = 970
OccupancyImageWidth = 256
OccupancyImageHeight = 512
imageOriginX = OccupancyImageWidth/2
imageOriginY = OccupancyImageHeight/2
lonResolution = OccupancyImageHeight/occupancyMapHeight
latResolution = OccupancyImageWidth/occupancyMapWidth
sampleSize = 100
inputSize = 50
outputSize = 100
imageFileType = '.png'
channel = 3
# This one is to pass some specific vehicle ids as list for situations like if lane change vehicles or turn vehicles.
LaneChangeVehicleList = [11.0,21.0,7.0,5.0,21.0,87.0,44.0,54.0,50.0,41.0,31.0,115.0,32.0,45.0,121.0,60.0,144.0]


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
            VehicleLength = vehicleVal[8]
            VehicleWidth = vehicleVal[9]
            VehicleSpeed = vehicleVal[11]
            VehicleTime = vehicleVal[3]
            frameData = dictionaryByFrames[str(FrameID)]
            frameSpecificList = []
            frameSpecificList.append([VehicleID, FrameID])
            frameSpecificList.append([VehicleX, VehicleY, VehicleLength, VehicleWidth, VehicleTime, VehicleSpeed])
            for frameVal in frameData:
                frameSpecificVehicleID = frameVal[0]
                currentTime = frameVal[3]
                if (frameSpecificVehicleID != VehicleID) and (VehicleTime==currentTime):
                    currentX = frameVal[4]
                    currentY = frameVal[5]
                    currentLength = frameVal[8]
                    currentWidth = frameVal[9]
                    currentSpeed = frameVal[11]
                    currentTime = frameVal[3]
                    frameSpecificList.append([currentX, currentY, currentLength, currentWidth, currentTime, currentSpeed])
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

    imageCount = 0
    concatinatedFrameArray = np.zeros((OccupancyImageHeight,OccupancyImageWidth,channel), dtype=np.uint8)
    concatinatedFrameArray.fill(255)
    cv2.namedWindow('test', cv2.WINDOW_NORMAL)

    for vehicle in vehiclePositions:
        currentVehicleID = vehicle[0][0][0]
        if(currentVehicleID not in LaneChangeVehicleList):
            continue
            
        for idx, _ in enumerate(vehicle):
            if (idx+outputSize+1 > len(vehicle)):
                break
            global folderCount
            print('Processing occupancy sample:' + str(folderCount) + ' for vehicleID ' + str(currentVehicleID))
            folderCount = folderCount + 1
            inputSampleFrames = vehicle[idx:idx+inputSize]
            localOriginX = inputSampleFrames[-1][1][0]
            localOriginY = inputSampleFrames[-1][1][1]
            inputPoseList = []
            for sampleFrame in inputSampleFrames:
                currentTargetX, currentTargetY, length, width, localTime, targetSpeed = sampleFrame[1]
                # Check if the Target car is out of the co ordinate frame
                absXDistance = abs(currentTargetX-localOriginX)
                absYDistance = abs(currentTargetY-localOriginY)
                if((absXDistance>(occupancyMapWidth/2))  or (absYDistance>(occupancyMapHeight/2))):
                    print('Target Car out of Frame')
                    sys.exit()
                relativeTargetX = imageOriginX + int((currentTargetX - localOriginX)*latResolution)
                relativeTargetY = imageOriginY - int((currentTargetY - localOriginY)*lonResolution)
                inputPoseList.append([relativeTargetX,relativeTargetY])
            
            pts = np.array(inputPoseList, np.int32)
            pts = pts.reshape((-1,1,2))
            cv2.polylines(concatinatedFrameArray,[pts],False,(0,0,255),1)
            
            outputSampleFrames = vehicle[idx+inputSize+1:idx+outputSize+1]
            outputPoseList = []
            for outputSample in outputSampleFrames:
                currentTargetX, currentTargetY, length, width, localTime, unusedSpeed = outputSample[1]
                relativeOutputTargetX = imageOriginX - int((currentTargetX - localOriginX)*latResolution)
                relativeOutputTargetY = imageOriginY - int((currentTargetY - localOriginY)*lonResolution)
                outputPoseList.append([relativeOutputTargetX,relativeOutputTargetY])
            
            pts = np.array(outputPoseList, np.int32)
            pts = pts.reshape((-1,1,2))
            cv2.polylines(concatinatedFrameArray,[pts],False,(255,0,0),1)
            
            cv2.imshow('test', concatinatedFrameArray)
            cv2.waitKey(1)

    cv2.imwrite('./Results/dist50.png',concatinatedFrameArray)
    
    print('All the occupancy grid map generated!!!!!!')


if __name__ == '__main__':


    # if(os.path.isdir(imageFolder)):
    #     print(imageFolder + ' folder exists. What to do with the current one??')
    #     sys.exit()
    # else:
    #     os.mkdir(imageFolder)

    testList=createRelativePositionList(trainingCsvFile)
    generateOccupancyGrid(testList)

