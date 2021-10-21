import sys
import csv
import numpy as np
from PIL import Image
import os
import shutil
import random

# This is to count how many samples created and also create new folder name each 
# time a new sample is created
folderCount = 0
# This is the absolute path for where the generated occupancy map will be saved
imageFolder = '/home/saptarshi/PythonCode/AdvanceLSTM/Joint/'
#laneChangeFolder = '/home/saptarshi/PythonCode/AdvanceLSTM/foldertest/LaneChange/'
#followRoadFolder = '/home/saptarshi/PythonCode/AdvanceLSTM/foldertest/FollowRoad/'
# This is the csv file absolute path for which the training OGM maps will be generated
trainingCsvFile = '/home/saptarshi/PythonCode/AdvanceLSTM/SplittedData/maptest.csv'

# Internal used variables:
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
imageFileType = '.png'
# This one is to pass some specific vehicle ids as list for situations like if lane change vehicles or turn vehicles.
LaneChangeVehicleList = [11.0,21.0,7.0,5.0,21.0,87.0,44.0,54.0,50.0,41.0,31.0,115.0,32.0,45.0,121.0,144.0]
normalVehicleCount = 25
for i in range(normalVehicleCount):
    LaneChangeVehicleList.append(random.randint(0,150))

FollowRoadStr = 'Follow Road'
LaneChangeStr = 'Lane Change'
AccelerationStr = 'Acceleration'
MaintainStr = 'Maintain'
DecelerationStr = 'Deceleration'
FollowRoadAccLabel      = '1,0,0,0,0,0'
FollowRoadMaintainLabel = '0,1,0,0,0,0'
FollowRoadDeccLabel     = '0,0,1,0,0,0'
LaneChageAccLabel       = '0,0,0,1,0,0'
LaneChageMaintainLabel  = '0,0,0,0,1,0'
LaneChageDeccLabel      = '0,0,0,0,0,1'

# Given the list of lane number for the output ground truth movement it decides
# if the sample is a lane change class or follow road class
def LaneBasedClassification(laneList):
    laneArray = np.array(laneList)
    laneCount = len(np.unique(laneArray))
    if(laneCount == 0):
        print('Lane number missing!!!')
        sys.exit()
    elif(laneCount == 1):
        laneClass = FollowRoadStr
    # If lane count is 2 count the number of frames for two different lanes.
    # Minumum 5 frames for each lane. So get the temporal window. Say in 30 frames the difference
    # should be <20 (25-5 = 20) but (28-2 = 26)
    elif(laneCount == 2):
        temporalWindow = outputSize-inputSize
        uniqueValues, occurCount = np.unique(laneArray, return_counts=True)
        if (occurCount[0]-occurCount[1] < 20):
            laneClass = LaneChangeStr
        else:
            laneClass = FollowRoadStr
    elif(laneCount == 3):
        laneClass = LaneChangeStr
    else:
        laneClass = LaneChangeStr
    
    return laneClass

def LongitudinalBasedClassification(inputMovement, outputMovement):
    ratio = outputMovement/inputMovement
    if (ratio<0.9):
        outputClass = DecelerationStr
    elif (ratio>=0.9 and ratio<=1.1):
        outputClass = MaintainStr
    elif(ratio > 1.1):
        outputClass = AccelerationStr
    else:
        print('Unknow class for longitudinal...')
        sys.exit()

    return outputClass


def CombineLateralAndLongitudinalLabel(laneClass,longClass):
    if(laneClass == FollowRoadStr):
        if(longClass == AccelerationStr):
            finalLabel = FollowRoadAccLabel
        elif(longClass == MaintainStr):
            finalLabel = FollowRoadMaintainLabel
        elif(longClass == DecelerationStr):
            finalLabel = FollowRoadDeccLabel
        else:
            print('Unknow Class during the combination')
            sys.exit()
    elif(laneClass == LaneChangeStr):
        if(longClass == AccelerationStr):
            finalLabel = LaneChageAccLabel
        elif(longClass == MaintainStr):
            finalLabel = LaneChageMaintainLabel
        elif(longClass == DecelerationStr):
            finalLabel = LaneChageDeccLabel
        else:
            print('Unknow Class during the combination')
            sys.exit()
    else:
        print('Unknow Class during the combination')
        sys.exit()
    return finalLabel


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
            VehicleTime = vehicleVal[3]
            TargetLaneNumber = vehicleVal[13]
            frameData = dictionaryByFrames[str(FrameID)]
            frameSpecificList = []
            frameSpecificList.append([VehicleID, FrameID])
            frameSpecificList.append([VehicleX, VehicleY, VehicleLength, VehicleWidth, VehicleTime, TargetLaneNumber])
            for frameVal in frameData:
                frameSpecificVehicleID = frameVal[0]
                currentTime = frameVal[3]
                if (frameSpecificVehicleID != VehicleID) and (VehicleTime==currentTime):
                    currentX = frameVal[4]
                    currentY = frameVal[5]
                    currentLength = frameVal[8]
                    currentWidth = frameVal[9]
                    currentTime = frameVal[3]
                    otherLaneNumber = frameVal[13]
                    frameSpecificList.append([currentX, currentY, currentLength, currentWidth, currentTime, otherLaneNumber])
            vehicleSpecificList.append(frameSpecificList)
        relativePositionList.append(vehicleSpecificList)
    
    loadFile.close()

    print('Relative Position List created!!!')

    return relativePositionList


# This one receives the previously explained list and creates the occupancy map samples.
#  Each sample is one folder under parent folder "imageFolder". Under each sample folder 
# there are 30 image frames which are the occupancy grid maps w.r.t one specific vehicle 
# for the last 30 frames and one "output·txt" file which holds the position of that specific
#  target car for next 30 frames w.r.t to last OGM frame in image co-ordinate system. 

def generateOccupancyGrid(vehiclePositions):

    print('Generating the Occupancy Grid Maps!!!')

    imageCount = 0
    laneWidthFeet = 14
    horizontaLimit = 45 #(2 car length 10*2=20 feet and gaps 5*2=10)

    for vehicle in vehiclePositions:
        currentVehicleID = vehicle[0][0][0]
        if(currentVehicleID not in LaneChangeVehicleList):
            continue
            
        for idx, _ in enumerate(vehicle):
            if (idx+outputSize+1 > len(vehicle)):
                break
            global folderCount
            folderPath = imageFolder + str(folderCount)
            os.mkdir(folderPath)
            print('Processing occupancy sample:' + str(folderCount) + ' for vehicleID ' + str(currentVehicleID))
            folderCount = folderCount + 1
            inputSampleFrames = vehicle[idx:idx+inputSize]
            localOriginX = inputSampleFrames[-1][1][0]
            localOriginY = inputSampleFrames[-1][1][1]
            inputInitialY = inputSampleFrames[0][1][1]
            inputLongitudinalMovement = abs(inputInitialY-localOriginY)
            for sampleFrame in inputSampleFrames:
                currentFrame = np.zeros((OccupancyImageHeight,OccupancyImageWidth), dtype=np.uint8)
                currentFrame.fill(255)
                currentTargetX, currentTargetY, length, width, localTime, targetLane = sampleFrame[1]
                # Check if the Target car is out of the co ordinate frame
                absXDistance = abs(currentTargetX-localOriginX)*latResolution
                absYDistance = abs(currentTargetY-localOriginY)*lonResolution
                if((absXDistance>=(occupancyMapWidth/2))  or (absYDistance>=(occupancyMapHeight/2))):
                    print('Target Car out of Frame')
                    sys.exit()
                relativeTargetX = imageOriginX + int((currentTargetX - localOriginX)*latResolution)
                relativeTargetY = imageOriginY - int((currentTargetY - localOriginY)*lonResolution)
                pixelLength = int(length*lonResolution)
                pixelWidth = int((width*latResolution)/2)
                currentFrame[int(relativeTargetY):int(relativeTargetY+pixelLength),int(relativeTargetX-pixelWidth):int(relativeTargetX+pixelWidth)] = 0
                frameLength = len(sampleFrame)
                for jdx in range(2,frameLength):
                    currentOtherX, currentOtherY, otherLength, OtherWidth, OtherLocalTime, OtherTargetLane = sampleFrame[jdx]
                    # Ignore other cars those are out of the co ordinate frame or far from the target car
                    absXDistance = abs(currentOtherX-currentTargetX)
                    absYDistance = abs(currentOtherY-currentTargetY)
                    if((absXDistance>(laneWidthFeet)) or (absYDistance>(horizontaLimit))):
                        continue
                    if (localTime == OtherLocalTime):
                        absoluteXPixel = imageOriginX + int((currentOtherX-localOriginX)*latResolution)
                        absoluteYPixel = imageOriginY - int((currentOtherY-localOriginY)*lonResolution)
                        pixelLength = int(otherLength*lonResolution)
                        pixelWidth = int((OtherWidth*latResolution)/2)
                        currentFrame[int(absoluteYPixel):int(absoluteYPixel+pixelLength),int(absoluteXPixel-pixelWidth):int(absoluteXPixel+pixelWidth)] = 128
                im = Image.fromarray(currentFrame)
                imageCount = imageCount+1
                imFileName = folderPath + '/' + str(imageCount) + imageFileType
                im.save(imFileName)
            
            outputSampleFrames = vehicle[idx+inputSize+1:idx+outputSize+1]
            outFileName = folderPath + '/' + 'output.txt'
            f = open(outFileName, 'w')
            laneNumberList = []
            outputInitialY = outputSampleFrames[0][1][1]
            outputFinalY = outputSampleFrames[-1][1][1]
            outputLongitudinalMovement = abs(outputInitialY-outputFinalY)

            for outputSample in outputSampleFrames:
                currentTargetX, currentTargetY, length, width, localTime, outputLane = outputSample[1]
                laneNumberList.append(outputLane)
                relativeOutputTargetX = imageOriginX + int((currentTargetX - localOriginX)*latResolution)
                relativeOutputTargetY = imageOriginY - int((currentTargetY - localOriginY)*lonResolution)
                outputString = str(relativeOutputTargetX) + ',' + str(relativeOutputTargetY) + '\n'
                f.writelines(outputString)

            # Check the lane change or follow class based on the lane number
            laneBasedClass = LaneBasedClassification(laneNumberList)
            # Check the longitudinal classification based on input and output movement
            longitudinalBasedClass = LongitudinalBasedClassification(inputLongitudinalMovement,outputLongitudinalMovement)
            # Combine the lateral and longitudinal label to formulate the final label
            finalLabel = CombineLateralAndLongitudinalLabel(laneBasedClass,longitudinalBasedClass)
            f.writelines(finalLabel)
            f.close()
            
    print('All the occupancy grid map generated!!!!!!')


if __name__ == '__main__':


    if(os.path.isdir(imageFolder)):
        print(imageFolder + ' folder exists. What to do with the current one??')
        sys.exit()
    else:
        os.mkdir(imageFolder)
        #os.mkdir(laneChangeFolder)
        #os.mkdir(followRoadFolder)

    testList=createRelativePositionList(trainingCsvFile)
    generateOccupancyGrid(testList)

