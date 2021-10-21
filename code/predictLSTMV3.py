import os
import PIL
from PIL import Image
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed, Conv2D, MaxPooling2D, Flatten
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
import cv2
import sys
import skimage
from keras import Model
import math
from scipy.io import savemat
import csv
import shutil

#K.set_image_dim_ordering('th')

#imageVisualTest = Image.new('RGB', (512, 128))
#imageVisualTest = np.empty((512, 128))
imageVisualTest = np.zeros((1024,128,3), np.uint8)
occupancyMapWidth = 100
occupancyMapHeight = 600
OccupancyImageWidth = 128
OccupancyImageHeight = 1024
imageOriginX = OccupancyImageWidth/2
imageOriginY = OccupancyImageHeight/2
lonResolution = OccupancyImageHeight/occupancyMapHeight
latResolution = OccupancyImageWidth/occupancyMapWidth

def euclidean_distance_loss(y_true, y_pred):
    """
    Euclidean distance loss
    https://en.wikipedia.org/wiki/Euclidean_distance
    :param y_true: TensorFlow/Theano tensor
    :param y_pred: TensorFlow/Theano tensor of the same shape as y_true
    :return: float
    """
    result = K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))
    return result


def LoadSamples():
    #Get all the folder of samples
    folderPath = '/home/saptarshi/PythonCode/AdvanceLSTM/TestData/test'
    folderList = []
    for i,j,y in sorted(os.walk(folderPath)):
        folderList.append(i)

    # Pop out the first item as this is holding the parent folder
    folderList.pop(0)

    allSample = []
    imageType = '.jpeg'
    outputFile = 'output.txt'
    for folder in folderList:
        eachSample = []
        sampleInput = []
        sampleOutput = []
        # Extract all the images for one sample as input
        fileList = os.listdir(folder)
        imagesList = [i for i in fileList if i.endswith(imageType)]
        sortedImageFiles = sorted(imagesList, key=lambda a: int(a.split(".")[0]) )
        for imageFile in sortedImageFiles:
            img = Image.open(folder + '/' +  imageFile)
            img.load()
            global imageVisualTest
            #imageVisualTest = Image.fromarray(np.asarray(img,dtype=np.uint8))
            imageVisualTest = np.asarray(img,dtype=np.uint8)
            imageData = np.asarray(img,dtype=np.uint8)/255
            sampleInput.append(imageData)
        # extract the future position from the output.txt file as output for each sample
        outputLines = open(folder + '/' +  outputFile,'r').read().splitlines()
        for outputValue in outputLines:
            outputX = float(outputValue.split(',')[0])
            outputY = float(outputValue.split(',')[1])
            sampleOutput.append([outputX,outputY])
        
        eachSample.append(sampleInput)
        eachSample.append(sampleOutput)
        allSample.append(eachSample)

    return allSample

def SplitTraininputOutput(samples):
    trainInput = []
    trainOutput = []
    for sample in samples:
        inputVal = sample[0]
        outputVal = np.array(sample[1])
        trainInput.append(inputVal)
        trainOutput.append(outputVal)
    trainInputArray = np.array(trainInput)
    trainOutputArray = np.array(trainOutput)
    return trainInputArray,trainOutputArray

def CreateVehicleAndFrameDict(loadFileName):

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

    loadFile.close()

    return dictionaryByFrames,dictionaryByVehicles


def PredictForAllCars(inputFileName):

    model = load_model('/home/saptarshi/PythonCode/AdvanceLSTM/TrainedModels/I80LaneData130Epochs.h5', custom_objects={'euclidean_distance_loss': euclidean_distance_loss})
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
            VehicleLength = vehicleVal[8]
            VehicleWidth = vehicleVal[9]
            VehicleTime = vehicleVal[3]
            frameData = dictByFrames[str(FrameID)]
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
                    frameSpecificList.append([currentX, currentY, currentLength, currentWidth, currentTime, frameSpecificVehicleID])
            vehicleSpecificList.append(frameSpecificList)
        relativePositionList.append(vehicleSpecificList)
    

    occupancyMapWidth = 100
    occupancyMapHeight = 600
    OccupancyImageWidth = 128
    OccupancyImageHeight = 512
    imageOriginX = OccupancyImageWidth/2
    imageOriginY = OccupancyImageHeight/2
    lonResolution = OccupancyImageHeight/occupancyMapHeight
    latResolution = OccupancyImageWidth/occupancyMapWidth
    imageCount = 0
    sampleSize = 60
    inputSize = 30
    outputSize = 60
    channel = 1
    temporal = 30
    imageFolder = '/home/saptarshi/PythonCode/AdvanceLSTM/PlaceHolderForPrediction/'
    resultFolder = '/home/saptarshi/PythonCode/AdvanceLSTM/Results/FrameResult/'
    globalResultFolder = '/home/saptarshi/PythonCode/AdvanceLSTM/Results/US101LaneChange/'
    targetCarFolder = 'target'
    otherCarFolder = 'other'
    imageFileType = '.png'
    outputCount = 0
    globalImageWidth = 400
    globalImageHeight = 3600
    globalFeetWidth = 100
    globalFeetHeight = 1800
    globalWidthResolution = globalImageWidth/globalFeetWidth
    globalHeightResolution = globalImageHeight/globalFeetHeight
    globalImage = np.zeros((globalImageHeight,globalImageWidth,3), np.uint8)
    globalImage.fill(255)

    #cv2.namedWindow('test', cv2.WINDOW_NORMAL)
    for vehicle in relativePositionList:
        trackerDict = dict()
        for idx, _ in enumerate(vehicle):
            if (idx >= len(vehicle)):
                break

            inputSampleFrames = vehicle[idx]
            primaryCarId = inputSampleFrames[0][0]
            currentTargetX, currentTargetY, length, width, localTime = inputSampleFrames[1]
            carsInCurrentFrame = []
            carsInCurrentFrame.append(primaryCarId)
            if primaryCarId in trackerDict:
                trackerDict[primaryCarId].append([currentTargetX, currentTargetY, length, width, localTime])
            else:
                trackerDict[primaryCarId] = [[currentTargetX, currentTargetY, length, width, localTime]]

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
                del trackerDict[str(death)]

            #check for target car 60 frame eligible check
            if(len(trackerDict[primaryCarId]) < 60):
                print('Tracker Data Under Populated!!!')

            if(len(trackerDict[primaryCarId]) > 60):
                print('Tracker Data Over Populated!!!')

            if ( len(trackerDict[primaryCarId]) == 60):
                os.mkdir(imageFolder + targetCarFolder)
                targetCar = trackerDict[primaryCarId]

                primaryLocalOriginX = targetCar[29][0]
                primaryLocalOriginY = targetCar[29][1]
                sample = 1
                primaryImageFiles = np.zeros((sample,temporal,OccupancyImageHeight,OccupancyImageWidth,channel))
            
                for ldx,sampleTargetFrame in enumerate(targetCar[0:30]):
                    currentFrame = np.zeros((OccupancyImageHeight,OccupancyImageWidth), dtype=np.uint8)
                    currentFrame.fill(255)
                    currentTargetX, currentTargetY, length, width, localTime = targetCar[ldx]
                    relativeTargetX = imageOriginX - int((currentTargetX - primaryLocalOriginX)*latResolution)
                    relativeTargetY = imageOriginY - int((currentTargetY - primaryLocalOriginY)*lonResolution)
                    pixelLength = int(length*lonResolution)
                    pixelWidth = int((width*latResolution)/2)
                    currentFrame[int(relativeTargetY):int(relativeTargetY+pixelLength),int(relativeTargetX-pixelWidth):int(relativeTargetX+pixelWidth)] = 0
                    for trackerKey in trackerDict:
                        if trackerKey!= primaryCarId and len(trackerDict[trackerKey]) > ldx and localTime == trackerDict[trackerKey][ldx][4]:
                            absoluteXPixel = imageOriginX - int((trackerDict[trackerKey][ldx][0]-primaryLocalOriginX)*latResolution)
                            absoluteYPixel = imageOriginY - int((trackerDict[trackerKey][ldx][1]-primaryLocalOriginY)*lonResolution)
                            pixelLength = int(trackerDict[trackerKey][ldx][2]*lonResolution)
                            pixelWidth = int((trackerDict[trackerKey][ldx][3]*latResolution)/2)
                            currentFrame[int(absoluteYPixel):int(absoluteYPixel+pixelLength),int(absoluteXPixel-pixelWidth):int(absoluteXPixel+pixelWidth)] = 128
                    im = Image.fromarray(currentFrame)
                    imageCount = imageCount+1
                    imFileName = imageFolder + targetCarFolder + '/' + str(imageCount) + imageFileType
                    primaryImageFiles[0,ldx,:,:,0] = im
                    im.save(imFileName)

                predictedPrimayPoses = model.predict(primaryImageFiles/255)

                opencvVisulaizeImage = cv2.cvtColor(currentFrame, cv2.COLOR_GRAY2BGR)

                #Plot the predicted for Target vehicle
                targetCarPoseList = []
                for predictedPose in predictedPrimayPoses[0]:
                    globalPoseX = (primaryLocalOriginX + ((imageOriginX-predictedPose[0])*(1/latResolution))) * globalWidthResolution
                    globalPoseY = (primaryLocalOriginY + ((imageOriginY-predictedPose[1])*(1/lonResolution))) * globalHeightResolution
                    targetCarPoseList.append([globalPoseX,globalPoseY])
                    cv2.circle(opencvVisulaizeImage,(predictedPose[0],predictedPose[1]), 2, (0,0,255))

                pts = np.array(targetCarPoseList, np.int32)
                pts = pts.reshape((-1,1,2))
                cv2.polylines(globalImage,[pts],False,(0,0,255),3)
                textPose = pts[0,0]
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(globalImage, str(primaryCarId), (textPose[0],textPose[1]), font, 1,(0,0,255),1,cv2.LINE_AA)

                #Plot the ground Truth for Target vehicle
                outputTargetSampleFrames = targetCar[30:59]
                targetCarGroundTruthPoseList = []
                for outputTargetSample in outputTargetSampleFrames:
                    currentTargetX, currentTargetY, length, width, localTime = outputTargetSample
                    relativeOutputTargetX = imageOriginX - int((currentTargetX - primaryLocalOriginX)*latResolution)
                    relativeOutputTargetY = imageOriginY - int((currentTargetY - primaryLocalOriginY)*lonResolution)
                    globalPoseX =  currentTargetX*globalWidthResolution
                    globalPoseY = currentTargetY*globalHeightResolution
                    targetCarGroundTruthPoseList.append([globalPoseX,globalPoseY])
                    cv2.circle(opencvVisulaizeImage,(int(relativeOutputTargetX),int(relativeOutputTargetY)), 2, (255,0,0))

                pts = np.array(targetCarGroundTruthPoseList, np.int32)
                pts = pts.reshape((-1,1,2))
                cv2.polylines(globalImage,[pts],False,(255,0,0),3)


                otherCarCount = 0
                for otherTrackedEligibleCarKey in trackerDict:
                    otherImageFiles = np.zeros((sample,temporal,OccupancyImageHeight,OccupancyImageWidth,channel))

                    if otherTrackedEligibleCarKey != primaryCarId and len(trackerDict[otherTrackedEligibleCarKey]) < 60:
                        print('Other Car ' + str(otherTrackedEligibleCarKey) + ' is not populated')

                    if otherTrackedEligibleCarKey != primaryCarId and len(trackerDict[otherTrackedEligibleCarKey]) > 60:
                        print('Other Car ' + str(otherTrackedEligibleCarKey) + ' is over populated')

                    if otherTrackedEligibleCarKey != primaryCarId and len(trackerDict[otherTrackedEligibleCarKey]) == 60:
                        otherCarCount = otherCarCount + 1
                        otherFolderPath = imageFolder + otherCarFolder + str(otherCarCount)
                        os.mkdir(otherFolderPath)
                        otherCarDataForPrediction = trackerDict[otherTrackedEligibleCarKey]
                        localOriginX = otherCarDataForPrediction[29][0]
                        localOriginY = otherCarDataForPrediction[29][1]

                        Xoffest = (localOriginX - primaryLocalOriginX)*latResolution
                        Yoffest = (localOriginY - primaryLocalOriginY)*lonResolution

                        for ndx,sampleOtherFrame in enumerate(otherCarDataForPrediction[0:30]):
                            currentFrame = np.zeros((OccupancyImageHeight,OccupancyImageWidth), dtype=np.uint8)
                            currentFrame.fill(255)
                            currentTargetX, currentTargetY, length, width, localTime, currentKey = otherCarDataForPrediction[ndx]
                            relativeTargetX = imageOriginX - int((currentTargetX - localOriginX)*latResolution)
                            relativeTargetY = imageOriginY - int((currentTargetY - localOriginY)*lonResolution)
                            pixelLength = int(length*lonResolution)
                            pixelWidth = int((width*latResolution)/2)
                            currentFrame[int(relativeTargetY):int(relativeTargetY+pixelLength),int(relativeTargetX-pixelWidth):int(relativeTargetX+pixelWidth)] = 0
                            for forOtherTrackerKey in trackerDict:
                                if forOtherTrackerKey!= currentKey and len(trackerDict[forOtherTrackerKey]) > ndx and localTime == trackerDict[forOtherTrackerKey][ndx][4]:
                                    absoluteXPixel = imageOriginX - int((trackerDict[forOtherTrackerKey][ndx][0]-localOriginX)*latResolution)
                                    absoluteYPixel = imageOriginY - int((trackerDict[forOtherTrackerKey][ndx][1]-localOriginY)*lonResolution)
                                    pixelLength = int(trackerDict[forOtherTrackerKey][ndx][2]*lonResolution)
                                    pixelWidth = int((trackerDict[forOtherTrackerKey][ndx][3]*latResolution)/2)
                                    currentFrame[int(absoluteYPixel):int(absoluteYPixel+pixelLength),int(absoluteXPixel-pixelWidth):int(absoluteXPixel+pixelWidth)] = 128
                            im = Image.fromarray(currentFrame)
                            imageCount = imageCount+1
                            imFileName = otherFolderPath + '/' + str(imageCount) + imageFileType
                            im.save(imFileName)
                            otherImageFiles[0,ndx,:,:,0] = im
                        
                        predictedOtherPoses = model.predict(otherImageFiles/255)

                        # Plot the predicted for other vehicles
                        globalPoseList = []
                        for predictedPose in predictedOtherPoses[0]:
                            localOtherX = predictedPose[0]-Xoffest
                            localOtherY = predictedPose[1]-Yoffest
                            globalPoseX = (primaryLocalOriginX + ((imageOriginX-localOtherX)*(1/latResolution))) * globalWidthResolution
                            globalPoseY = (primaryLocalOriginY + ((imageOriginY-localOtherY)*(1/lonResolution))) * globalHeightResolution
                            globalPoseList.append([globalPoseX,globalPoseY])
                            cv2.circle(opencvVisulaizeImage,(int(predictedPose[0]-Xoffest),int(predictedPose[1]-Yoffest)), 2, (0,0,255))

                        pts = np.array(globalPoseList, np.int32)
                        pts = pts.reshape((-1,1,2))
                        cv2.polylines(globalImage,[pts],False,(0,0,255),3)
                        textPose = pts[0,0]
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(globalImage, str(otherTrackedEligibleCarKey), (textPose[0],textPose[1]), font, 1,(0,0,255),1,cv2.LINE_AA)

                        #Plot the ground Truth for other vehicles
                        otherCarGroundTruthPoseList = []
                        outputOtherSampleFrames = otherCarDataForPrediction[30:59]
                        for outputOtherSample in outputOtherSampleFrames:
                            currentTargetX, currentTargetY, length, width, localTime, currentKey = outputOtherSample
                            relativeOtherX = imageOriginX - int((currentTargetX - localOriginX)*latResolution)
                            relativeOtherY = imageOriginY - int((currentTargetY - localOriginY)*lonResolution)
                            offsetOtherX = relativeOtherX - Xoffest
                            offsetOtherY = relativeOtherY - Yoffest
                            globalPoseX =  currentTargetX*globalWidthResolution
                            globalPoseY = currentTargetY*globalHeightResolution
                            otherCarGroundTruthPoseList.append([globalPoseX,globalPoseY])
                            cv2.circle(opencvVisulaizeImage,(int(offsetOtherX),int(offsetOtherY)), 2, (255,0,0))

                        pts = np.array(otherCarGroundTruthPoseList, np.int32)
                        pts = pts.reshape((-1,1,2))
                        cv2.polylines(globalImage,[pts],False,(255,0,0),3)

                        #cv2.imshow('test', opencvVisulaizeImage)
                        #cv2.waitKey(1)
                        # outputFileName = resultFolder + str(outputCount) + imageFileType
                        # outputCount = outputCount + 1
                        # cv2.imwrite(outputFileName, opencvVisulaizeImage)
                        trackerDict[otherTrackedEligibleCarKey].pop(0)
                
                outputFileName = resultFolder + str(outputCount) + imageFileType
                cv2.imwrite(outputFileName, opencvVisulaizeImage)
                globalOutputFileName = globalResultFolder + str(outputCount) + imageFileType
                rotated90 = np.rot90(globalImage)
                cv2.imwrite(globalOutputFileName, rotated90)
                outputCount = outputCount + 1
                #cv2.imshow('test',rotated90)
                #cv2.waitKey(1)
                # Remove the older car Trajectories
                globalImage.fill(255)
                # Remove all the folders..
                removeFolders = os.listdir(imageFolder)
                for rmDir in removeFolders:
                    shutil.rmtree(imageFolder + rmDir)


                trackerDict[primaryCarId].pop(0)

    print('All the occupancy grid map generated!!!!!!')


if __name__ == '__main__':

    #PredictForAllCars('/home/saptarshi/PythonCode/AdvanceLSTM/SplittedData/trajectories-0400-0415.csv')
    PredictForAllCars('/home/saptarshi/Documents/US-101-LosAngeles-CA/vehicle-trajectory-data/0820am-0835am/trajectories-0820am-0835am.csv')

    sys.exit()

    # testList=createRelativePositionList('/home/saptarshi/PythonCode/AdvanceLSTM/SplittedData/maptest.csv')

    #model = load_model('TrainedModels/AdvanceLSTMV1.h5')
    model = load_model('/home/saptarshi/PythonCode/AdvanceLSTM/TrainedModels/AdvanceLSTMV6.h5', custom_objects={'euclidean_distance_loss': euclidean_distance_loss})
    model.summary()
    sys.exit()

    trainSamples = LoadSamples()
    numberOfSamples = len(trainSamples)
    trainX,trainY = SplitTraininputOutput(trainSamples)
    trainX = trainX.reshape(numberOfSamples,30,512,128,1)
    predicted = model.predict(trainX)
    groundTruth = np.reshape(trainY,(30,2))
    predictedReshaped = np.reshape(predicted,(30,2))

    if(len(groundTruth) != len(predictedReshaped)):
        print('Mismatched shape for ground truth and Predicted')
        sys.exit()

    predictionLength = len(predictedReshaped)
    
    #cv2.namedWindow('test', cv2.WINDOW_NORMAL)

    #global imageVisualTest
    #colorcv = cv2.cvtColor(imageVisualTest,cv2.COLOR_GRAY2RGB)
    # resizedIMage = imageVisualTest.resize((256, 1024), PIL.Image.ANTIALIAS)
    # opencvImage = np.array(resizedIMage) 

    #cv2.line(imageVisualTest,(0,0),(511,511),(255,0,0),1)
    #cv2.imshow('test', imageVisualTest)
    #cv2.waitKey(0)

    groundTruthPosXLast = int(groundTruth[0][0])
    groundTruthPosYLast = int(groundTruth[0][1])
    predictedPosXLast = int(predictedReshaped[0][0])
    predictedPosYLast = int(predictedReshaped[0][1])
    distErrorList = []

    for idx in range(1,predictionLength):
        groundTruthPosX = int(groundTruth[idx][0])
        groundTruthPosY = int(groundTruth[idx][1])
        predictedPosX = int(predictedReshaped[idx][0])
        predictedPosY = int(predictedReshaped[idx][1])
        diffX = pow(((groundTruthPosX-predictedPosX)*(1/latResolution)),2)
        diffY = pow(((groundTruthPosY-predictedPosY)*(1/lonResolution)),2)
        distError = math.sqrt(diffX+diffY)*0.3048
        distErrorList.append(distError)
        cv2.line((0,0,255), (groundTruthPosXLast,groundTruthPosYLast), (groundTruthPosX,groundTruthPosY), (255,0,0), 1)
        cv2.line((0,0,255),(predictedPosXLast,predictedPosYLast), (predictedPosX,predictedPosY), (0,0,255), 1)
        groundTruthPosXLast = groundTruthPosX
        groundTruthPosYLast = groundTruthPosY
        predictedPosXLast = predictedPosX
        predictedPosYLast = predictedPosY

    pilImage = Image.fromarray((0,0,255))
    resizedPILIMage = pilImage.resize((512, 2048), PIL.Image.ANTIALIAS)
    opencvImage = np.array(resizedPILIMage)
    #resizedIMage = imageVisualTest.resize((256, 1024), PIL.Image.ANTIALIAS)
    #resized = cv2.resize(imageVisualTest, (512, 2048), interpolation = cv2.INTER_CUBIC)
    #opencvImage = np.array(resizedIMage) 
    #cv2.imwrite('test.png', opencvImage)
    #cv2.waitKey(0)

    with open('res.csv', "w") as f:
        writer = csv.writer(f)
        for row in distErrorList:
            writer.writerow(row)
    #plt.plot(distErrorList)
    #plt.show()

    print('done....')








