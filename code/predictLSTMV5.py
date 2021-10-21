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
import tensorflow as tf
import random

# def euclidean_distance_loss(y_true, y_pred):
#     """
#     Euclidean distance loss
#     https://en.wikipedia.org/wiki/Euclidean_distance
#     :param y_true: TensorFlow/Theano tensor
#     :param y_pred: TensorFlow/Theano tensor of the same shape as y_true
#     :return: float
#     """
#     result = K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))
#     return result


# Specify the test trajectory csv file
testTrajFilePath = '/home/saptarshi/PythonCode/AdvanceLSTM/SplittedData/i80-0400-0415.csv'
# Load the pre trained model
modelFilePath = '/home/saptarshi/PythonCode/AdvanceLSTM/TrainedModels/velocity1.h5'
#model = load_model(modelFilePath, custom_objects={'EuclidianLoss': EuclidianLoss, 'EuclidianDistanceMetric' : EuclidianDistanceMetric})
# Set the Intermediate place holder image folder where it will save the image frames temporarily 
# before doing the prediction and clean the folder afterwards.
imageFolder = '/home/saptarshi/PythonCode/AdvanceLSTM/PlaceHolderForPrediction/'
# Remove any folders from the last if remaining from the place holder folder.
removeFolders = os.listdir(imageFolder)
for rmDir in removeFolders:
    shutil.rmtree(imageFolder + rmDir)

# Set the Folder for individual frame predicition results
resultFolder = '/home/saptarshi/PythonCode/AdvanceLSTM/Results/FrameResult/'
# Set the Folder for final global results (The predicted position plotted on the global scene)
globalResultFolder = '/home/saptarshi/PythonCode/AdvanceLSTM/Results/velocity/'

# Randomly select vehicles to do the test
TestVehicleList = []
testVehicleCount = 50
for i in range(testVehicleCount):
    TestVehicleList.append(random.randint(0,200))

# Set the different Occupancy Grid map and scene dimensions
occupancyMapWidth = 100
occupancyMapHeight = 600
OccupancyImageWidth = 256
OccupancyImageHeight = 512
imageOriginX = OccupancyImageWidth/2
imageOriginY = OccupancyImageHeight/2
lonResolution = OccupancyImageHeight/occupancyMapHeight 
latResolution = OccupancyImageWidth/occupancyMapWidth
inputSize = 30
outputSize = 60
channel = 2
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
globalLatErrorCount = 0
globalLonErrorCount = 0

# # Custome Euclidian distance loss function
def CustomeLoss(y_true, y_pred):
    alpha = tf.constant(0.65)
    latLoss = tf.abs(y_true[:,:,0] - y_pred[:,:,0])
    #latLoss = tf.Print(latLoss, [latLoss], 'latLoss:', summarize=1000)
    lonLoss = tf.abs(y_true[:,:,1] - y_pred[:,:,1])
    modifiedLatLoss = tf.multiply(latLoss,alpha)
    modifiedLonLoss = tf.multiply(lonLoss,(1-alpha))
    finalLoss = modifiedLatLoss + modifiedLonLoss
    return finalLoss

# # Orignial Euclidian Distance Error Metric
def EuclidianDistanceMetric(y_true, y_pred):
    latError = (y_true[:,:,0] - y_pred[:,:,0]) * (y_true[:,:,0] - y_pred[:,:,0])
    lonError = (y_true[:,:,1] - y_pred[:,:,1]) * (y_true[:,:,1] - y_pred[:,:,1])
    totalError = tf.reduce_mean(tf.sqrt(latError[:,:] + lonError[:,:]))
    return totalError


# Original Euclidian distance loss function (not ready....)
def EuclidianLoss(y_true, y_pred):
    alpha = tf.constant(0.65)
    lonResolutionTFConstant = tf.constant(lonResolution)
    latResolutionTFConstant = tf.constant(latResolution)
    latLossEuclidian = tf.multiply(tf.square(y_true[:,:,0] - y_pred[:,:,0]), latResolutionTFConstant)
    lonLossEuclidian = tf.multiply(tf.square(y_true[:,:,1] - y_pred[:,:,1]), lonResolutionTFConstant)
    modifiedLatLoss = tf.multiply(latLossEuclidian,alpha)
    modifiedLonLoss = tf.multiply(lonLossEuclidian,(1-alpha))
    finalLoss = tf.sqrt(modifiedLatLoss + modifiedLonLoss)
    return finalLoss

def euclidean_distance_loss(y_true, y_pred):
    """
    Euclidean distance loss
    https://en.wikipedia.org/wiki/Euclidean_distance
    :param y_true: TensorFlow/Theano tensor
    :param y_pred: TensorFlow/Theano tensor of the same shape as y_true
    :return: float
    """
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

# Due to wrong combination
model = load_model(modelFilePath, custom_objects={'euclidean_distance_loss': euclidean_distance_loss, 'euclidean_distance_loss' : euclidean_distance_loss})

# Calculate the average position error for each 30 frame sequences
def PositionError(y_true, y_pred):
    error = 0
    for jdx,val in enumerate(y_true):
        diffX = pow(((y_true[jdx,0]-y_pred[jdx,0])*(1/latResolution)),2)
        diffY = pow(((y_true[jdx,1]-y_pred[jdx,1])*(1/lonResolution)),2)
        distError = math.sqrt(diffX+diffY)*0.3048
        error = error + distError
    if (error<0):
        print('error is : ' + str(error))
    return error/30

# Calculate the average position error corresponding to the future frame count
def FrameError(y_true, y_pred):
    errorList = []
    for jdx,val in enumerate(y_true):
        diffX = pow(((y_true[jdx,0]-y_pred[jdx,0])*(1/latResolution)*feetToMeter),2)
        diffY = pow(((y_true[jdx,1]-y_pred[jdx,1])*(1/lonResolution)*feetToMeter),2)
        distError = math.sqrt(diffX+diffY)
        errorList.append(distError)
    if (distError<0):
        print('error is : ' + str(distError))
    global globalErrorCount
    globalErrorCount = globalErrorCount + 1
    return np.array(errorList)

# Calculate the average lateral position error corresponding to the future frame count
def FrameLatError(y_true, y_pred):
    errorList = []
    for jdx,val in enumerate(y_true):
        diffX = abs((y_true[jdx,0]-y_pred[jdx,0])*(1/latResolution)*feetToMeter)
        errorList.append(diffX)
    if (diffX<0):
        print('error is : ' + str(diffX))
    global globalLatErrorCount
    globalLatErrorCount = globalLatErrorCount + 1
    return np.array(errorList)

# Calculate the average longitudinal position error corresponding to the future frame count
def FrameLonError(y_true, y_pred):
    errorList = []
    for jdx,val in enumerate(y_true):
        diffY = abs((y_true[jdx,1]-y_pred[jdx,1])*(1/lonResolution)*feetToMeter)
        errorList.append(diffY)
    if (diffY<0):
        print('error is : ' + str(diffY))
    global globalLonErrorCount
    globalLonErrorCount = globalLonErrorCount + 1
    return np.array(errorList)

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

# Perfrom the predicition for all the cars in the scene and plot them in the global frame and save in the specified folder.
def PredictForAllCars(inputFileName):

    totalCurrentError = np.zeros(temporal)
    totalLatError = np.zeros(temporal)
    totalLonError = np.zeros(temporal)
    imageCount = 0
    outputCount = 0
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
    

    globalImage = np.zeros((globalImageHeight,globalImageWidth,3), np.uint8)
    globalImage.fill(255)

    #cv2.namedWindow('test', cv2.WINDOW_NORMAL)
    for vehicle in relativePositionList:
        currentVehicleID = vehicle[0][0][0]
        if((currentVehicleID not in TestVehicleList) and (globalErrorCount < 2000)):
            continue
        trackerDict = dict()
        print('Predicting for vehicle ID: ' + str(vehicle[0][0][0]))
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
                del trackerDict[death]

            #check for target car 60 frame eligible check
            # if(len(trackerDict[primaryCarId]) < 60):
            #     print('Tracker Data Under Populated!!!')

            # if(len(trackerDict[primaryCarId]) > 60):
            #     print('Tracker Data Over Populated!!!')

            if ( len(trackerDict[primaryCarId]) == outputSize):
                os.mkdir(imageFolder + targetCarFolder)
                targetCar = trackerDict[primaryCarId]

                primaryLocalOriginX = targetCar[inputSize-1][0]
                primaryLocalOriginY = targetCar[inputSize-1][1]
                sample = 1
                primaryImageFiles = np.zeros((sample,temporal,OccupancyImageHeight,OccupancyImageWidth,channel))
            
                for ldx,sampleTargetFrame in enumerate(targetCar[0:inputSize]):
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
                outputTargetSampleFrames = targetCar[inputSize:outputSize]
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

                # Calculate the frame based error
                groundTruthForError = np.array(outputTargetSampleFrames)[:,0:2]
                totalCurrentError = totalCurrentError + FrameError(np.array(targetCarPoseList), np.array(targetCarGroundTruthPoseList))
                totalLatError = totalLatError + FrameLatError(np.array(targetCarPoseList), np.array(targetCarGroundTruthPoseList))
                totalLonError = totalLonError + FrameLonError(np.array(targetCarPoseList), np.array(targetCarGroundTruthPoseList))


                otherCarCount = 0
                for otherTrackedEligibleCarKey in trackerDict:
                    otherImageFiles = np.zeros((sample,temporal,OccupancyImageHeight,OccupancyImageWidth,channel))

                    # if otherTrackedEligibleCarKey != primaryCarId and len(trackerDict[otherTrackedEligibleCarKey]) < 60:
                    #     print('Other Car ' + str(otherTrackedEligibleCarKey) + ' is not populated')

                    # if otherTrackedEligibleCarKey != primaryCarId and len(trackerDict[otherTrackedEligibleCarKey]) > 60:
                    #     print('Other Car ' + str(otherTrackedEligibleCarKey) + ' is over populated')

                    if otherTrackedEligibleCarKey != primaryCarId and len(trackerDict[otherTrackedEligibleCarKey]) == outputSize:
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
                        outputOtherSampleFrames = otherCarDataForPrediction[inputSize:outputSize]
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

                        totalCurrentError = totalCurrentError + FrameError(np.array(globalPoseList), np.array(otherCarGroundTruthPoseList))
                        totalLatError = totalLatError + FrameLatError(np.array(globalPoseList), np.array(otherCarGroundTruthPoseList))
                        totalLonError = totalLonError + FrameLonError(np.array(globalPoseList), np.array(otherCarGroundTruthPoseList))

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
    print(str(globalErrorCount) + ' number of sampls predicted...')
    plt.plot(totalCurrentError/globalErrorCount, label='Total Error')
    plt.plot(totalLatError/globalLatErrorCount, label='Lateral Error')
    plt.plot(totalLonError/globalLonErrorCount, label='Longitudinal Error')
    plt.legend()
    plt.savefig('US101.png')
    plt.show()


if __name__ == '__main__':

    PredictForAllCars(testTrajFilePath)

    print('All the cars are prediect in the scene.')

    sys.exit()







