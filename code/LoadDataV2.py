import sys
import csv
import numpy as np
from PIL import Image
import os

folderCount = 0

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
            VehicleLength = vehicleVal[8]
            VehicleWidth = vehicleVal[9]
            VehicleTime = vehicleVal[3]
            frameData = dictionaryByFrames[str(FrameID)]
            frameSpecificList = []
            frameSpecificList.append([VehicleID, FrameID])
            frameSpecificList.append([VehicleX, VehicleY, VehicleLength, VehicleWidth, VehicleTime])
            for frameVal in frameData:
                frameSpecificVehicleID = frameVal[0]
                currentTime = frameVal[3]
                if (frameSpecificVehicleID != VehicleID) and (VehicleTime==currentTime):
                    currentX = frameVal[4]
                    currentY = frameVal[5]
                    currentLength = frameVal[8]
                    currentWidth = frameVal[9]
                    currentTime = frameVal[3]
                    frameSpecificList.append([currentX, currentY, currentLength, currentWidth, currentTime])
            vehicleSpecificList.append(frameSpecificList)
        relativePositionList.append(vehicleSpecificList)
    
    loadFile.close()

    print('Relative Position List created for one file')

    return relativePositionList

def generateOccupancyGrid(vehiclePositions):

    print('Generating the Occupancy Grid Maps')

    occupancyMapWidth = 100
    occupancyMapHeight = 600
    OccupancyImageWidth = 128
    OccupancyImageHeight = 1024
    imageOriginX = OccupancyImageWidth/2
    imageOriginY = OccupancyImageHeight/2
    lonResolution = OccupancyImageHeight/occupancyMapHeight
    latResolution = OccupancyImageWidth/occupancyMapWidth
    imageCount = 0
    sampleSize = 60
    inputSize = 30
    outputSize = 60
    imageFolder = '/home/saptarshi/PythonCode/AdvanceLSTM/us-101TestData/'
    imageFileType = '.png'
    for vehicle in vehiclePositions:
        for idx, _ in enumerate(vehicle):
            if (idx+outputSize > len(vehicle)):
                break
            global folderCount
            folderPath = imageFolder + str(folderCount)
            os.mkdir(folderPath)
            print('Processing occupancy sample:' + str(folderCount))
            folderCount = folderCount + 1
            inputSampleFrames = vehicle[idx:idx+inputSize]
            localOriginX = inputSampleFrames[-1][1][0]
            localOriginY = inputSampleFrames[-1][1][1]
            for sampleFrame in inputSampleFrames:
                currentFrame = np.zeros((OccupancyImageHeight,OccupancyImageWidth), dtype=np.uint8)
                currentFrame.fill(255)
                currentTargetX, currentTargetY, length, width, localTime = sampleFrame[1]
                relativeTargetX = imageOriginX - int((currentTargetX - localOriginX)*latResolution)
                relativeTargetY = imageOriginY - int((currentTargetY - localOriginY)*lonResolution)
                pixelLength = int(length*lonResolution)
                pixelWidth = int((width*latResolution)/2)
                currentFrame[int(relativeTargetY):int(relativeTargetY+pixelLength),int(relativeTargetX-pixelWidth):int(relativeTargetX+pixelWidth)] = 0
                frameLength = len(sampleFrame)
                for jdx in range(2,frameLength):
                    if (localTime == sampleFrame[jdx][4]):
                        absoluteXPixel = imageOriginX - int((sampleFrame[jdx][0]-localOriginX)*latResolution)
                        absoluteYPixel = imageOriginY - int((sampleFrame[jdx][1]-localOriginY)*lonResolution)
                        pixelLength = int(sampleFrame[jdx][2]*lonResolution)
                        pixelWidth = int((sampleFrame[jdx][3]*latResolution)/2)
                        currentFrame[int(absoluteYPixel):int(absoluteYPixel+pixelLength),int(absoluteXPixel-pixelWidth):int(absoluteXPixel+pixelWidth)] = 128
                im = Image.fromarray(currentFrame)
                imageCount = imageCount+1
                imFileName = folderPath + '/' + str(imageCount) + imageFileType
                im.save(imFileName)
            
            outputSampleFrames = vehicle[idx+inputSize+1:idx+outputSize+1]
            outFileName = folderPath + '/' + 'output.txt'
            f = open(outFileName, 'w')
            for outputSample in outputSampleFrames:
                currentTargetX, currentTargetY, length, width, localTime = outputSample[1]
                relativeOutputTargetX = imageOriginX - int((currentTargetX - localOriginX)*latResolution)
                relativeOutputTargetY = imageOriginY - int((currentTargetY - localOriginY)*lonResolution)
                outputString = str(relativeOutputTargetX) + ',' + str(relativeOutputTargetY) + '\n'
                f.writelines(outputString)
            f.close()
            
    print('All the occupancy grid map generated!!!!!!')


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

    #testList=createRelativePositionList('/home/saptarshi/Documents/I-80-Emeryville-CA/vehicle-trajectory-data/0400pm-0415pm/trajectories-0400-0415.csv')
    testList=createRelativePositionList('/home/saptarshi/PythonCode/AdvanceLSTM/SplittedData/trajectories-0400-0415.csv')
    #testList=createRelativePositionList('/home/saptarshi/PythonCode/AdvanceLSTM/SplittedData/bug.csv')
    generateOccupancyGrid(testList)

