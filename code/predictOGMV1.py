import os


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
 
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="0"


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
from pyproj import Proj, transform
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
modelFilePath = '/home/saptarshi/PythonCode/AdvanceLSTM/BestModels/ConcatNoMaxPoolBestCheck.h5'
#model = load_model(modelFilePath, custom_objects={'EuclidianLoss': EuclidianLoss, 'EuclidianDistanceMetric' : EuclidianDistanceMetric})
# Set the Intermediate place holder image folder where it will save the image frames temporarily 
# before doing the prediction and clean the folder afterwards.
imageFolder = '/home/saptarshi/PythonCode/AdvanceLSTM/PlaceHolderForPrediction/'
# Remove any folders from the last if remaining from the place holder folder.
removeFolders = os.listdir(imageFolder)
for rmDir in removeFolders:
    shutil.rmtree(imageFolder + rmDir)

# Set the Folder for individual frame predicition results
resultFolder = '/home/saptarshi/PythonCode/AdvanceLSTM/Results/Check/'
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
OccupancyImageWidth = 128
OccupancyImageHeight = 1024
imageOriginX = OccupancyImageWidth/2
imageOriginY = OccupancyImageHeight/2
lonResolution = OccupancyImageHeight/occupancyMapHeight 
latResolution = OccupancyImageWidth/occupancyMapWidth
inputSize = 30
outputSize = 80
predTemporal = outputSize-inputSize
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
globalLatErrorCount = 0
globalLonErrorCount = 0
observeWidth = 40
observeLength = 70

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
model.summary()

#print(model.get_weights())

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
    return error/temporal

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
    print('Global Error Count = ' + str(globalErrorCount))
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
            VehicleSpeed = vehicleVal[11]
            frameData = dictByFrames[str(FrameID)]
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
                    currentTime = frameVal[3]
                    currentSpeed = frameVal[11]
                    frameSpecificList.append([currentX, currentY, currentLength, currentWidth, currentTime, frameSpecificVehicleID, currentSpeed])
            vehicleSpecificList.append(frameSpecificList)
        relativePositionList.append(vehicleSpecificList)
    

    globalImage = np.zeros((globalImageHeight,globalImageWidth,3), np.uint8)
    globalImage.fill(255)

    #cv2.namedWindow('test', cv2.WINDOW_NORMAL)
    for vehicle in relativePositionList:
        currentVehicleID = vehicle[0][0][0]
        if(currentVehicleID not in TestVehicleList):
           continue
        trackerDict = dict()
        print('Predicting for vehicle ID: ' + str(vehicle[0][0][0]))
        for idx, _ in enumerate(vehicle):
            if (idx >= len(vehicle)):
                break

            inputSampleFrames = vehicle[idx]
            primaryCarId = inputSampleFrames[0][0]
            currentTargetX, currentTargetY, length, width, localTime, targetSpeed = inputSampleFrames[1]
            carsInCurrentFrame = []
            carsInCurrentFrame.append(primaryCarId)
            if primaryCarId in trackerDict:
                trackerDict[primaryCarId].append([currentTargetX, currentTargetY, length, width, localTime, targetSpeed])
            else:
                trackerDict[primaryCarId] = [[currentTargetX, currentTargetY, length, width, localTime, targetSpeed]]

            frameLength = len(inputSampleFrames)
            for jdx in range(2,frameLength):
                currentOtherX, currentOtherY, otherLength, otherWidth, otherLocalTime, otherCarId, otherSpeed = inputSampleFrames[jdx]
                if (localTime == otherLocalTime):
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
                    concatinatedFrameArray = np.zeros((OccupancyImageHeight,OccupancyImageWidth,channel), dtype=np.float)
                    currentFrame = np.zeros((OccupancyImageHeight,OccupancyImageWidth), dtype=np.uint8)
                    currentFrame.fill(255)
                    #velocityFrame = np.zeros((OccupancyImageHeight,OccupancyImageWidth), dtype=np.float)
                    #velocityFrame.fill(0)
                    currentTargetX, currentTargetY, length, width, localTime, targetSpeed = targetCar[ldx]
                    relativeTargetX = imageOriginX + int((currentTargetX - primaryLocalOriginX)*latResolution)
                    relativeTargetY = imageOriginY - int((currentTargetY - primaryLocalOriginY)*lonResolution)
                    pixelLength = int(length*lonResolution)
                    pixelWidth = int((width*latResolution)/2)
                    currentFrame[int(relativeTargetY):int(relativeTargetY+pixelLength),int(relativeTargetX-pixelWidth):int(relativeTargetX+pixelWidth)] = 0
                    #velocityFrame[int(relativeTargetY):int(relativeTargetY+pixelLength),int(relativeTargetX-pixelWidth):int(relativeTargetX+pixelWidth)] = targetSpeed
                    for trackerKey in trackerDict:
                        if trackerKey!= primaryCarId and len(trackerDict[trackerKey]) > ldx and localTime == trackerDict[trackerKey][ldx][4]:
                            otherSpeed = trackerDict[trackerKey][ldx][6]
                            absoluteXPixel = imageOriginX + int((trackerDict[trackerKey][ldx][0]-primaryLocalOriginX)*latResolution)
                            absoluteYPixel = imageOriginY - int((trackerDict[trackerKey][ldx][1]-primaryLocalOriginY)*lonResolution)
                            pixelLength = int(trackerDict[trackerKey][ldx][2]*lonResolution)
                            pixelWidth = int((trackerDict[trackerKey][ldx][3]*latResolution)/2)
                            currentFrame[int(absoluteYPixel):int(absoluteYPixel+pixelLength),int(absoluteXPixel-pixelWidth):int(absoluteXPixel+pixelWidth)] = 128
                            #velocityFrame[int(absoluteYPixel):int(absoluteYPixel+pixelLength),int(absoluteXPixel-pixelWidth):int(absoluteXPixel+pixelWidth)] = otherSpeed
                    im = Image.fromarray(currentFrame)
                    imageCount = imageCount+1
                    imFileName = imageFolder + targetCarFolder + '/' + str(imageCount) + imageFileType
                    primaryImageFiles[0,ldx,:,:,0] = currentFrame/255
                    #primaryImageFiles[0,ldx,:,:,1] = velocityFrame/100
                    im.save(imFileName)

                predictedPrimayPoses = model.predict(primaryImageFiles)

                opencvVisulaizeImage = cv2.cvtColor(currentFrame, cv2.COLOR_GRAY2BGR)

                #Plot the predicted for Target vehicle
                targetCarPoseList = []
                for predictedPose in predictedPrimayPoses[0]:
                    globalPoseX = (primaryLocalOriginX + ((imageOriginX+predictedPose[0])*(1/latResolution))) * globalWidthResolution
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
                    currentTargetX, currentTargetY, length, width, localTime, unusedSpeed = outputTargetSample
                    relativeOutputTargetX = imageOriginX + int((currentTargetX - primaryLocalOriginX)*latResolution)
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

                # Need a looooooooooot changes from here...........................


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
                        localOriginX = otherCarDataForPrediction[temporal-1][0]
                        localOriginY = otherCarDataForPrediction[temporal-1][1]

                        Xoffest = (localOriginX - primaryLocalOriginX)*latResolution
                        Yoffest = (localOriginY - primaryLocalOriginY)*lonResolution

                        for ndx,sampleOtherFrame in enumerate(otherCarDataForPrediction[0:temporal]):
                            concatinatedFrameArray = np.zeros((OccupancyImageHeight,OccupancyImageWidth,channel), dtype=np.float)
                            currentFrame = np.zeros((OccupancyImageHeight,OccupancyImageWidth), dtype=np.uint8)
                            currentFrame.fill(255)
                            #otherVelocityFrame = np.zeros((OccupancyImageHeight,OccupancyImageWidth), dtype=np.float)
                            #otherVelocityFrame.fill(0)
                            currentTargetX, currentTargetY, length, width, localTime, currentKey, otherSpeed = otherCarDataForPrediction[ndx]
                            relativeTargetX = imageOriginX + int((currentTargetX - localOriginX)*latResolution)
                            relativeTargetY = imageOriginY - int((currentTargetY - localOriginY)*lonResolution)
                            pixelLength = int(length*lonResolution)
                            pixelWidth = int((width*latResolution)/2)
                            currentFrame[int(relativeTargetY):int(relativeTargetY+pixelLength),int(relativeTargetX-pixelWidth):int(relativeTargetX+pixelWidth)] = 0
                            #otherVelocityFrame[int(relativeTargetY):int(relativeTargetY+pixelLength),int(relativeTargetX-pixelWidth):int(relativeTargetX+pixelWidth)] = otherSpeed
                            for forOtherTrackerKey in trackerDict:
                                if forOtherTrackerKey!= currentKey and len(trackerDict[forOtherTrackerKey]) > ndx and localTime == trackerDict[forOtherTrackerKey][ndx][4]:
                                    if forOtherTrackerKey == primaryCarId:
                                        otherSpeed = trackerDict[forOtherTrackerKey][ndx][5]
                                    else:
                                        otherSpeed = trackerDict[forOtherTrackerKey][ndx][6]
                                    absoluteXPixel = imageOriginX + int((trackerDict[forOtherTrackerKey][ndx][0]-localOriginX)*latResolution)
                                    absoluteYPixel = imageOriginY - int((trackerDict[forOtherTrackerKey][ndx][1]-localOriginY)*lonResolution)
                                    pixelLength = int(trackerDict[forOtherTrackerKey][ndx][2]*lonResolution)
                                    pixelWidth = int((trackerDict[forOtherTrackerKey][ndx][3]*latResolution)/2)
                                    currentFrame[int(absoluteYPixel):int(absoluteYPixel+pixelLength),int(absoluteXPixel-pixelWidth):int(absoluteXPixel+pixelWidth)] = 128
                                    #otherVelocityFrame[int(absoluteYPixel):int(absoluteYPixel+pixelLength),int(absoluteXPixel-pixelWidth):int(absoluteXPixel+pixelWidth)] = otherSpeed

                            #im = Image.fromarray(concatinatedFrameArray)
                            imageCount = imageCount+1
                            imFileName = otherFolderPath + '/' + str(imageCount) + imageFileType
                            #im.save(imFileName)
                            otherImageFiles[0,ndx,:,:,0] = currentFrame/255
                            #otherImageFiles[0,ndx,:,:,1] = otherVelocityFrame/100
                        
                        predictedOtherPoses = model.predict(otherImageFiles)

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
                            currentTargetX, currentTargetY, length, width, localTime, currentKey, unusedSpeed = outputOtherSample
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
    plt.savefig('oldtest.png')
    plt.show()


# Perfrom the predicition for all the cars in the scene and plot them in the global frame and save in the specified folder.
def PredictForTargetCars(inputFileName):

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
    lonError = np.zeros(predTemporal)

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
            VehicleSpeed = vehicleVal[11]
            frameData = dictByFrames[str(FrameID)]
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
                    currentTime = frameVal[3]
                    currentSpeed = frameVal[11]
                    frameSpecificList.append([currentX, currentY, currentLength, currentWidth, currentTime, frameSpecificVehicleID, currentSpeed])
            vehicleSpecificList.append(frameSpecificList)
        relativePositionList.append(vehicleSpecificList)
    

    globalImage = np.zeros((globalImageHeight,globalImageWidth,3), np.uint8)
    globalImage.fill(255)

    #cv2.namedWindow('test', cv2.WINDOW_NORMAL)
    for vehicle in relativePositionList:
        currentVehicleID = vehicle[0][0][0]
        if(currentVehicleID not in TestVehicleList):
           continue
        trackerDict = dict()
        print('Predicting for vehicle ID: ' + str(vehicle[0][0][0]))
        for idx, _ in enumerate(vehicle):
            if (idx >= len(vehicle)):
                break

            inputSampleFrames = vehicle[idx]
            primaryCarId = inputSampleFrames[0][0]
            currentTargetX, currentTargetY, length, width, localTime, targetSpeed = inputSampleFrames[1]
            carsInCurrentFrame = []
            carsInCurrentFrame.append(primaryCarId)
            if primaryCarId in trackerDict:
                trackerDict[primaryCarId].append([currentTargetX, currentTargetY, length, width, localTime, targetSpeed])
            else:
                trackerDict[primaryCarId] = [[currentTargetX, currentTargetY, length, width, localTime, targetSpeed]]

            frameLength = len(inputSampleFrames)
            for jdx in range(2,frameLength):
                currentOtherX, currentOtherY, otherLength, otherWidth, otherLocalTime, otherCarId, otherSpeed = inputSampleFrames[jdx]
                if (localTime == otherLocalTime):
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

                primaryLocalOriginXOriginal = targetCar[inputSize-1][0]
                primaryLocalOriginYOriginal = targetCar[inputSize-1][1]


                sample = 1
                primaryImageFiles = np.zeros((sample,temporal,OccupancyImageHeight,OccupancyImageWidth,channel))

                targetCarImagePoseList = []
                #otherCarsPixelDict = dict() 

                pixelLengthTarget = 0
                pixelWidthTarget = 0
            
                for ldx,sampleTargetFrame in enumerate(targetCar[0:inputSize]):
                    #concatinatedFrameArray = np.zeros((OccupancyImageHeight,OccupancyImageWidth,channel), dtype=np.float)
                    currentFrame = np.zeros((OccupancyImageHeight,OccupancyImageWidth), dtype=np.uint8)
                    currentFrame.fill(255)
                    # velocityFrame = np.zeros((OccupancyImageHeight,OccupancyImageWidth), dtype=np.float)
                    # velocityFrame.fill(0)
                    currentTargetX, currentTargetY, length, width, localTime, targetSpeed = targetCar[ldx]
                    # Check if the target car in within the frame
                    absXDistance = abs(currentTargetX-primaryLocalOriginX)
                    absYDistance = abs(currentTargetY-primaryLocalOriginY)
                    if((absXDistance>=(occupancyMapWidth/2))  or (absYDistance>=(occupancyMapHeight/2))):
                        print('Target Car out of Frame')
                        sys.exit()

                    relativeTargetX = imageOriginX + int((currentTargetX - primaryLocalOriginX)*latResolution)
                    relativeTargetY = imageOriginY - int((currentTargetY - primaryLocalOriginY)*lonResolution)
                    targetCarImagePoseList.append([relativeTargetX, relativeTargetY])
                    pixelLengthTarget = int(length*lonResolution)
                    pixelWidthTarget = int((width*latResolution)/2)
                    #currentFrame[int(relativeTargetY),int(relativeTargetX)] = 0
                    currentFrame[int(relativeTargetY):int(relativeTargetY+pixelLengthTarget),int(relativeTargetX-pixelWidthTarget):int(relativeTargetX+pixelWidthTarget)] = 0
                    for trackerKey in trackerDict:
                        # Extract the coresponding Index in the list for the other car
                        corespndingOtherCarIndex = len(trackerDict[trackerKey]) - outputSize + ldx
                        if corespndingOtherCarIndex >= 0 and trackerKey!= primaryCarId and localTime == trackerDict[trackerKey][corespndingOtherCarIndex][4]:
                            otherSpeed = trackerDict[trackerKey][corespndingOtherCarIndex][6]
                            currentOtherXCheck = trackerDict[trackerKey][corespndingOtherCarIndex][0]
                            currentOtherYCheck = trackerDict[trackerKey][corespndingOtherCarIndex][1]
                            # Ignore other cars those are out of the co ordinate frame
                            absXDistance = abs(currentOtherXCheck-primaryLocalOriginX)
                            absYDistance = abs(currentOtherYCheck-primaryLocalOriginY)
                            if((absXDistance>=(occupancyMapWidth/2)) or (absYDistance>=(occupancyMapHeight/2))):
                                continue

                            absoluteXPixel = imageOriginX + int((currentOtherXCheck-primaryLocalOriginX)*latResolution)
                            absoluteYPixel = imageOriginY - int((currentOtherYCheck-primaryLocalOriginY)*lonResolution)
                            # # Add in the dictionary for later relative pose use
                            # if trackerKey in otherCarsPixelDict:
                            #     otherCarsPixelDict[trackerKey].append([absoluteXPixel,absoluteYPixel])
                            # else:
                            #     otherCarsPixelDict[trackerKey] = [[absoluteXPixel,absoluteYPixel]]
                            pixelLength = int(trackerDict[trackerKey][corespndingOtherCarIndex][2]*lonResolution)
                            pixelWidth = int((trackerDict[trackerKey][corespndingOtherCarIndex][3]*latResolution)/2)
                            #currentFrame[int(absoluteYPixel),int(absoluteXPixel)] = 128
                            currentFrame[int(absoluteYPixel):int(absoluteYPixel+pixelLength),int(absoluteXPixel-pixelWidth):int(absoluteXPixel+pixelWidth)] = 128
                    im = Image.fromarray(currentFrame)
                    imageCount = imageCount+1
                    imFileName = imageFolder + targetCarFolder + '/' + str(imageCount) + imageFileType
                    primaryImageFiles[0,ldx,:,:,0] = currentFrame/255
                    #primaryImageFiles[0,ldx,:,:,1] = velocityFrame/100
                    im.save(imFileName)

                targetCarPredictedPoses = []


                for t in range(1,predTemporal+1):

                    opencvVisulaizeImage = cv2.cvtColor(currentFrame, cv2.COLOR_GRAY2BGR)

                    # Predict the next pose in Image frame
                    predictedPrimayPoses = model.predict(primaryImageFiles)

                    # Remove the folder after prediction
                    removeFolders = os.listdir(imageFolder)
                    for rmDir in removeFolders:
                        shutil.rmtree(imageFolder + rmDir)
                    
                    # recreate the target forlders
                    os.mkdir(imageFolder + targetCarFolder)

                    # Collect the Next frame prediction
                    predictedPoseX =  predictedPrimayPoses[0][0]
                    predictedPoseY =  predictedPrimayPoses[0][1]

                    if predictedPoseY > OccupancyImageHeight/2:
                        print('In the wrong Update')
                        predictedPoseY = (OccupancyImageHeight/2) - 1 ##- 2

                    # Calculate the shift from the last position
                    shiftX = (OccupancyImageWidth/2) - predictedPoseX
                    shiftY = (OccupancyImageHeight/2) - predictedPoseY

                    if not targetCarPredictedPoses:
                        targetCarPredictedPoses.append([predictedPoseX,predictedPoseY])
                    else:
                        lastPredX = targetCarPredictedPoses[-1][0]
                        lastPredY = targetCarPredictedPoses[-1][1]
                        newPredX = lastPredX - shiftX
                        newPredY = lastPredY - shiftY
                        targetCarPredictedPoses.append([newPredX,newPredY])

                    # Extract the first and second item to estimate the shift
                    firstX = targetCarImagePoseList[0][0]
                    firstY = targetCarImagePoseList[0][1]
                    secondX = targetCarImagePoseList[1][0]
                    secondY = targetCarImagePoseList[1][1]
                    newShiftX = firstX - secondX
                    newShiftY = firstY - secondY

                    targetCarImagePoseList = targetCarImagePoseList[1:]
                    targetCarImagePoseList.append([predictedPoseX,predictedPoseY])
                    # Update all the positions based on the new prediction 
                    for udx, _ in enumerate(targetCarImagePoseList):
                        targetCarImagePoseList[udx][0] = targetCarImagePoseList[udx][0] + newShiftX
                        targetCarImagePoseList[udx][1] = targetCarImagePoseList[udx][1] + newShiftY

                    # Update the origins
                    primaryLocalOriginX = primaryLocalOriginX + (shiftX*(1/latResolution))
                    primaryLocalOriginY = primaryLocalOriginY + (shiftY*(1/lonResolution))

                    primaryImageFiles = np.zeros((sample,temporal,OccupancyImageHeight,OccupancyImageWidth,channel))

                    for vdx in range(0,temporal):
                        currentFrame = np.zeros((OccupancyImageHeight,OccupancyImageWidth), dtype=np.uint8)
                        currentFrame.fill(255)

                        # Draw the target car
                        relativeTargetX = targetCarImagePoseList[vdx][0]
                        relativeTargetY = targetCarImagePoseList[vdx][1]
                        currentFrame[int(relativeTargetY):int(relativeTargetY+pixelLengthTarget),int(relativeTargetX-pixelWidthTarget):int(relativeTargetX+pixelWidthTarget)] = 0

                        # Draw Other cars


                        for trackerKey in trackerDict:
                            # Extract the coresponding Index in the list for the other car
                            shiftedTimeIndex = t + vdx
                            localTime = targetCar[shiftedTimeIndex][4]
                            corespndingOtherCarIndex = len(trackerDict[trackerKey]) - outputSize + shiftedTimeIndex
                            if corespndingOtherCarIndex >= 0 and trackerKey!= primaryCarId and localTime == trackerDict[trackerKey][corespndingOtherCarIndex][4]:
                                otherSpeed = trackerDict[trackerKey][corespndingOtherCarIndex][6]
                                currentOtherXCheck = trackerDict[trackerKey][corespndingOtherCarIndex][0]
                                currentOtherYCheck = trackerDict[trackerKey][corespndingOtherCarIndex][1]
                                # Ignore other cars those are out of the co ordinate frame
                                absXDistance = abs(currentOtherXCheck-primaryLocalOriginX)
                                absYDistance = abs(currentOtherYCheck-primaryLocalOriginY)
                                if((absXDistance>=(occupancyMapWidth/2)) or (absYDistance>=(occupancyMapHeight/2))):
                                    continue

                                # For some weired check
                                currentOriginX = targetCar[inputSize + t - 1][0]
                                currentOriginY = targetCar[inputSize + t - 1][1]
                                absoluteXPixel = imageOriginX + int((currentOtherXCheck-currentOriginX)*latResolution)
                                absoluteYPixel = imageOriginY - int((currentOtherYCheck-currentOriginY)*lonResolution)

                                #if(corespndingOtherCarIndex == 0):
                                #if(trackerKey not in otherCarsPixelDict):
                                # # # absoluteXPixel = imageOriginX + int((currentOtherXCheck-primaryLocalOriginX)*latResolution)
                                # # # absoluteYPixel = imageOriginY - int((currentOtherYCheck-primaryLocalOriginY)*lonResolution)
                                #     otherCarsPixelDict[trackerKey] = [[absoluteXPixel,absoluteYPixel]]
                                # else:
                                #     currentOtherXCheckOld = trackerDict[trackerKey][corespndingOtherCarIndex-1][0]
                                #     currentOtherYCheckOld = trackerDict[trackerKey][corespndingOtherCarIndex-1][1]
                                #     pxShiftX = int((currentOtherXCheckOld - currentOtherXCheck)*latResolution)
                                #     pxShiftY = int((currentOtherYCheckOld - currentOtherYCheck)*lonResolution)
                                #     # Get the last pixel from the other dict and subtract the shift to get current and append the current pixel pose
                                #     lastXPixel = otherCarsPixelDict[trackerKey][-1][0]
                                #     lastYPixel = otherCarsPixelDict[trackerKey][-1][1]
                                #     absoluteXPixel = lastXPixel + pxShiftX
                                #     absoluteYPixel = lastYPixel + pxShiftY
                                #     #otherCarsPixelDict[trackerKey].pop(0)
                                #     otherCarsPixelDict[trackerKey].append([int(absoluteXPixel),int(absoluteYPixel)])


                                pixelLength = int(trackerDict[trackerKey][corespndingOtherCarIndex][2]*lonResolution)
                                pixelWidth = int((trackerDict[trackerKey][corespndingOtherCarIndex][3]*latResolution)/2)
                                #currentFrame[int(absoluteYPixel),int(absoluteXPixel)] = 128
                                currentFrame[int(absoluteYPixel):int(absoluteYPixel+pixelLength),int(absoluteXPixel-pixelWidth):int(absoluteXPixel+pixelWidth)] = 128

                    
                        im = Image.fromarray(currentFrame)
                        imageCount = imageCount+1
                        imFileName = imageFolder + targetCarFolder + '/' + str(imageCount) + imageFileType
                        primaryImageFiles[0,vdx,:,:,0] = currentFrame/255
                        #primaryImageFiles[0,ldx,:,:,1] = velocityFrame/100
                        im.save(imFileName)



                # Extract the ground truth
                outputTargetSampleFrames = targetCar[inputSize:outputSize]
                targetCarGroundTruthPoseList = []
                for outputTargetSample in outputTargetSampleFrames:
                    currentTargetX, currentTargetY, length, width, localTime, unusedSpeed = outputTargetSample
                    relativeOutputTargetX = imageOriginX + int((currentTargetX - primaryLocalOriginXOriginal)*latResolution)
                    relativeOutputTargetY = imageOriginY - int((currentTargetY - primaryLocalOriginYOriginal)*lonResolution)
                    targetCarGroundTruthPoseList.append([relativeOutputTargetX,relativeOutputTargetY])

                # Plot the ground Truth
                pts = np.array(targetCarGroundTruthPoseList, np.int32)
                pts = pts.reshape((-1,1,2))
                cv2.polylines(opencvVisulaizeImage,[pts],False,(0,255,0),3)
                textPose = pts[0,0]
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(opencvVisulaizeImage, str(primaryCarId), (textPose[0],textPose[1]), font, 0.5,(0,0,255),1,cv2.LINE_AA)

                # Plot the Predicted 
                pts = np.array(targetCarPredictedPoses, np.int32)
                pts = pts.reshape((-1,1,2))
                cv2.polylines(opencvVisulaizeImage,[pts],False,(255,0,0),3)
                textPose = pts[0,0]
                font = cv2.FONT_HERSHEY_SIMPLEX

                rotated = np.rot90(opencvVisulaizeImage)
                rotated = np.rot90(rotated)
                rotated = np.rot90(rotated)

                cv2.imwrite(resultFolder + str(globalLonErrorCount) + '.png', rotated)



                # Calculate the frame based error
                # groundTruthForError = np.array(outputTargetSampleFrames)[:,0:2]
                # totalCurrentError = totalCurrentError + FrameError(np.array(targetCarPoseList), np.array(targetCarGroundTruthPoseList))
                # totalLatError = totalLatError + FrameLatError(np.array(targetCarPoseList), np.array(targetCarGroundTruthPoseList))
                # totalLonError = totalLonError + FrameLonError(np.array(targetCarPoseList), np.array(targetCarGroundTruthPoseList))

                lonError = lonError + FrameLonError(np.array(targetCarPredictedPoses), np.array(targetCarGroundTruthPoseList))

                print('Predicted Poses \n')

                print(targetCarPredictedPoses)

                print('Ground Truth Poses \n')
                print(targetCarGroundTruthPoseList)

                print('Error count ' + str(globalLonErrorCount))

                print(lonError/globalLonErrorCount)

                
                # # Remove all the folders..
                removeFolders = os.listdir(imageFolder)
                for rmDir in removeFolders:
                    shutil.rmtree(imageFolder + rmDir)


                trackerDict[primaryCarId].pop(0)
                for trackerKey in trackerDict:
                    trackLength = len(trackerDict[trackerKey])
                    if trackLength >= outputSize and trackerKey!= primaryCarId:
                        trackerDict[trackerKey].pop(0)





    print(str(globalErrorCount) + ' number of sampls predicted...')
    print('Frame Total Errors')
    print(totalCurrentError/globalErrorCount)
    print('Frame Lon Errors')
    print(totalLonError/globalErrorCount)
    print('Frame Lat Errors')
    print(totalLatError/globalErrorCount)
    plt.plot(totalCurrentError/globalErrorCount, label='Total Error')
    plt.plot(totalLatError/globalLatErrorCount, label='Lateral Error')
    plt.plot(totalLonError/globalLonErrorCount, label='Longitudinal Error')
    plt.xlabel('Frame Count', fontsize=15)
    plt.ylabel('Error (meter)', fontsize=15)
    plt.legend()
    plt.savefig('loss1.29.png')
    plt.show()








    # Perfrom the predicition for all the cars in the scene and plot them in the global frame and save in the specified folder.
def PredictForAllCarsNew(inputFileName):

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
    lonError = np.zeros(predTemporal)

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
            VehicleSpeed = vehicleVal[11]
            frameData = dictByFrames[str(FrameID)]
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
                    currentTime = frameVal[3]
                    currentSpeed = frameVal[11]
                    frameSpecificList.append([currentX, currentY, currentLength, currentWidth, currentTime, frameSpecificVehicleID, currentSpeed])
            vehicleSpecificList.append(frameSpecificList)
        relativePositionList.append(vehicleSpecificList)
    

    globalImage = np.zeros((globalImageHeight,globalImageWidth,3), np.uint8)
    globalImage.fill(255)

    #cv2.namedWindow('test', cv2.WINDOW_NORMAL)
    for vehicle in relativePositionList:
        currentVehicleID = vehicle[0][0][0]
        if(currentVehicleID not in TestVehicleList):
           continue
        trackerDict = dict()
        print('Predicting for vehicle ID: ' + str(vehicle[0][0][0]))
        for idx, _ in enumerate(vehicle):
            if (idx >= len(vehicle)):
                break

            inputSampleFrames = vehicle[idx]
            primaryCarId = inputSampleFrames[0][0]
            currentTargetX, currentTargetY, length, width, localTime, targetSpeed = inputSampleFrames[1]
            carsInCurrentFrame = []
            carsInCurrentFrame.append(primaryCarId)
            if primaryCarId in trackerDict:
                trackerDict[primaryCarId].append([currentTargetX, currentTargetY, length, width, localTime, targetSpeed])
            else:
                trackerDict[primaryCarId] = [[currentTargetX, currentTargetY, length, width, localTime, targetSpeed]]

            frameLength = len(inputSampleFrames)
            for jdx in range(2,frameLength):
                currentOtherX, currentOtherY, otherLength, otherWidth, otherLocalTime, otherCarId, otherSpeed = inputSampleFrames[jdx]
                if (localTime == otherLocalTime):
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

            if(len(trackerDict[primaryCarId]) > outputSize):
                print('Tracker Data Over Populated!!!')
                sys.exit()

            if ( len(trackerDict[primaryCarId]) == outputSize):
                os.mkdir(imageFolder + targetCarFolder)
                targetCar = trackerDict[primaryCarId]

                primaryLocalOriginX = targetCar[inputSize-1][0]
                primaryLocalOriginY = targetCar[inputSize-1][1]

                primaryLocalOriginXOriginal = targetCar[inputSize-1][0]
                primaryLocalOriginYOriginal = targetCar[inputSize-1][1]


                sample = 1
                primaryImageFiles = np.zeros((sample,temporal,OccupancyImageHeight,OccupancyImageWidth,channel))

                targetCarImagePoseList = []
                pixelLengthTarget = 0
                pixelWidthTarget = 0
            
                for ldx,sampleTargetFrame in enumerate(targetCar[0:inputSize]):
                    #concatinatedFrameArray = np.zeros((OccupancyImageHeight,OccupancyImageWidth,channel), dtype=np.float)
                    currentFrame = np.zeros((OccupancyImageHeight,OccupancyImageWidth), dtype=np.uint8)
                    currentFrame.fill(255)
                    # velocityFrame = np.zeros((OccupancyImageHeight,OccupancyImageWidth), dtype=np.float)
                    # velocityFrame.fill(0)
                    currentTargetX, currentTargetY, length, width, localTime, targetSpeed = targetCar[ldx]
                    # Check if the target car in within the frame
                    absXDistance = abs(currentTargetX-primaryLocalOriginX)
                    absYDistance = abs(currentTargetY-primaryLocalOriginY)
                    if((absXDistance>=(occupancyMapWidth/2))  or (absYDistance>=(occupancyMapHeight/2))):
                        print('Target Car out of Frame')
                        sys.exit()

                    relativeTargetX = imageOriginX + int((currentTargetX - primaryLocalOriginX)*latResolution)
                    relativeTargetY = imageOriginY - int((currentTargetY - primaryLocalOriginY)*lonResolution)
                    targetCarImagePoseList.append([relativeTargetX, relativeTargetY])
                    pixelLengthTarget = int(length*lonResolution)
                    pixelWidthTarget = int((width*latResolution)/2)
                    #currentFrame[int(relativeTargetY),int(relativeTargetX)] = 0
                    currentFrame[int(relativeTargetY):int(relativeTargetY+pixelLengthTarget),int(relativeTargetX-pixelWidthTarget):int(relativeTargetX+pixelWidthTarget)] = 0
                    for trackerKey in trackerDict:
                        # Extract the coresponding Index in the list for the other car
                        corespndingOtherCarIndex = len(trackerDict[trackerKey]) - outputSize + ldx
                        if corespndingOtherCarIndex >= 0 and trackerKey!= primaryCarId and localTime == trackerDict[trackerKey][corespndingOtherCarIndex][4]:
                            otherSpeed = trackerDict[trackerKey][corespndingOtherCarIndex][6]
                            currentOtherXCheck = trackerDict[trackerKey][corespndingOtherCarIndex][0]
                            currentOtherYCheck = trackerDict[trackerKey][corespndingOtherCarIndex][1]
                            # Ignore other cars those are out of the co ordinate frame
                            absXDistance = abs(currentOtherXCheck-primaryLocalOriginX)
                            absYDistance = abs(currentOtherYCheck-primaryLocalOriginY)
                            if((absXDistance>=(occupancyMapWidth/2)) or (absYDistance>=(occupancyMapHeight/2))):
                                continue

                            absoluteXPixel = imageOriginX + int((currentOtherXCheck-primaryLocalOriginX)*latResolution)
                            absoluteYPixel = imageOriginY - int((currentOtherYCheck-primaryLocalOriginY)*lonResolution)
                            pixelLength = int(trackerDict[trackerKey][corespndingOtherCarIndex][2]*lonResolution)
                            pixelWidth = int((trackerDict[trackerKey][corespndingOtherCarIndex][3]*latResolution)/2)
                            #currentFrame[int(absoluteYPixel),int(absoluteXPixel)] = 128
                            currentFrame[int(absoluteYPixel):int(absoluteYPixel+pixelLength),int(absoluteXPixel-pixelWidth):int(absoluteXPixel+pixelWidth)] = 128
                    # im = Image.fromarray(currentFrame)
                    # imageCount = imageCount+1
                    # imFileName = imageFolder + targetCarFolder + '/' + str(imageCount) + imageFileType
                    primaryImageFiles[0,ldx,:,:,0] = currentFrame/255
                    #primaryImageFiles[0,ldx,:,:,1] = velocityFrame/100
                    # im.save(imFileName)

                opencvVisulaizeImage = cv2.cvtColor(currentFrame, cv2.COLOR_GRAY2BGR)

                targetCarPredictedPoses = []


                for t in range(1,predTemporal+1):

                    # Predict the next pose in Image frame
                    predictedPrimayPoses = model.predict(primaryImageFiles)

                    # Remove the folder after prediction
                    removeFolders = os.listdir(imageFolder)
                    for rmDir in removeFolders:
                        shutil.rmtree(imageFolder + rmDir)
                    
                    # recreate the target forlders
                    os.mkdir(imageFolder + targetCarFolder)

                    # Collect the Next frame prediction
                    predictedPoseX =  predictedPrimayPoses[0][0]
                    predictedPoseY =  predictedPrimayPoses[0][1]

                    if predictedPoseY > OccupancyImageHeight/2:
                        print('In the wrong Update')
                        predictedPoseY = (OccupancyImageHeight/2) - 1 ##- 2

                    # Calculate the shift from the last position
                    shiftX = (OccupancyImageWidth/2) - predictedPoseX
                    shiftY = (OccupancyImageHeight/2) - predictedPoseY

                    if not targetCarPredictedPoses:
                        targetCarPredictedPoses.append([predictedPoseX,predictedPoseY])
                    else:
                        lastPredX = targetCarPredictedPoses[-1][0]
                        lastPredY = targetCarPredictedPoses[-1][1]
                        newPredX = lastPredX +  shiftX  #Something weird check   - shiftX
                        newPredY = lastPredY - shiftY
                        targetCarPredictedPoses.append([newPredX,newPredY])

                    # Extract the first and second item to estimate the shift
                    firstX = targetCarImagePoseList[0][0]
                    firstY = targetCarImagePoseList[0][1]
                    secondX = targetCarImagePoseList[1][0]
                    secondY = targetCarImagePoseList[1][1]
                    newShiftX = firstX - secondX
                    newShiftY = firstY - secondY

                    targetCarImagePoseList = targetCarImagePoseList[1:]
                    targetCarImagePoseList.append([predictedPoseX,predictedPoseY])
                    # Update all the positions based on the new prediction 
                    for udx, _ in enumerate(targetCarImagePoseList):
                        targetCarImagePoseList[udx][0] = targetCarImagePoseList[udx][0] + newShiftX
                        targetCarImagePoseList[udx][1] = targetCarImagePoseList[udx][1] + newShiftY

                    # Update the origins
                    primaryLocalOriginX = primaryLocalOriginX + (shiftX*(1/latResolution))
                    primaryLocalOriginY = primaryLocalOriginY + (shiftY*(1/lonResolution))

                    primaryImageFiles = np.zeros((sample,temporal,OccupancyImageHeight,OccupancyImageWidth,channel))

                    for vdx in range(0,temporal):
                        currentFrame = np.zeros((OccupancyImageHeight,OccupancyImageWidth), dtype=np.uint8)
                        currentFrame.fill(255)

                        # Draw the target car
                        relativeTargetX = targetCarImagePoseList[vdx][0]
                        relativeTargetY = targetCarImagePoseList[vdx][1]
                        currentFrame[int(relativeTargetY):int(relativeTargetY+pixelLengthTarget),int(relativeTargetX-pixelWidthTarget):int(relativeTargetX+pixelWidthTarget)] = 0

                        # Draw Other cars
                        for trackerKey in trackerDict:
                            # Extract the coresponding Index in the list for the other car
                            shiftedTimeIndex = t + vdx
                            localTime = targetCar[shiftedTimeIndex][4]
                            corespndingOtherCarIndex = len(trackerDict[trackerKey]) - outputSize + shiftedTimeIndex
                            if corespndingOtherCarIndex >= 0 and trackerKey!= primaryCarId and localTime == trackerDict[trackerKey][corespndingOtherCarIndex][4]:
                                otherSpeed = trackerDict[trackerKey][corespndingOtherCarIndex][6]
                                currentOtherXCheck = trackerDict[trackerKey][corespndingOtherCarIndex][0]
                                currentOtherYCheck = trackerDict[trackerKey][corespndingOtherCarIndex][1]
                                # Ignore other cars those are out of the co ordinate frame
                                absXDistance = abs(currentOtherXCheck-primaryLocalOriginX)
                                absYDistance = abs(currentOtherYCheck-primaryLocalOriginY)
                                if((absXDistance>=(occupancyMapWidth/2)) or (absYDistance>=(occupancyMapHeight/2))):
                                    continue

                                absoluteXPixel = imageOriginX + int((currentOtherXCheck-primaryLocalOriginX)*latResolution)
                                absoluteYPixel = imageOriginY - int((currentOtherYCheck-primaryLocalOriginY)*lonResolution)
                                pixelLength = int(trackerDict[trackerKey][corespndingOtherCarIndex][2]*lonResolution)
                                pixelWidth = int((trackerDict[trackerKey][corespndingOtherCarIndex][3]*latResolution)/2)
                                #currentFrame[int(absoluteYPixel),int(absoluteXPixel)] = 128
                                currentFrame[int(absoluteYPixel):int(absoluteYPixel+pixelLength),int(absoluteXPixel-pixelWidth):int(absoluteXPixel+pixelWidth)] = 128

                    
                        # im = Image.fromarray(currentFrame)
                        # imageCount = imageCount+1
                        # imFileName = imageFolder + targetCarFolder + '/' + str(imageCount) + imageFileType
                        primaryImageFiles[0,vdx,:,:,0] = currentFrame/255
                        #primaryImageFiles[0,ldx,:,:,1] = velocityFrame/100
                        # im.save(imFileName)



                # Extract the ground truth
                outputTargetSampleFrames = targetCar[inputSize:outputSize]
                targetCarGroundTruthPoseList = []
                for outputTargetSample in outputTargetSampleFrames:
                    currentTargetX, currentTargetY, length, width, localTime, unusedSpeed = outputTargetSample
                    relativeOutputTargetX = imageOriginX + int((currentTargetX - primaryLocalOriginXOriginal)*latResolution)
                    relativeOutputTargetY = imageOriginY - int((currentTargetY - primaryLocalOriginYOriginal)*lonResolution)
                    targetCarGroundTruthPoseList.append([relativeOutputTargetX,relativeOutputTargetY])

                # Plot the ground Truth
                pts = np.array(targetCarGroundTruthPoseList, np.int32)
                pts = pts.reshape((-1,1,2))
                cv2.polylines(opencvVisulaizeImage,[pts],False,(0,255,0),2)
                textPose = pts[0,0]
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(opencvVisulaizeImage, str(int(primaryCarId)), (textPose[0],textPose[1]), font, 0.4,(0,0,255),1,cv2.LINE_AA)

                # Plot the Predicted 
                pts = np.array(targetCarPredictedPoses, np.int32)
                pts = pts.reshape((-1,1,2))
                cv2.polylines(opencvVisulaizeImage,[pts],False,(255,0,0),2)
                textPose = pts[0,0]
                font = cv2.FONT_HERSHEY_SIMPLEX

                # rotated = np.rot90(opencvVisulaizeImage)
                # rotated = np.rot90(rotated)
                # rotated =np.rot90(rotated)

                # cv2.imwrite(resultFolder + str(globalLonErrorCount) + '.png', rotated)



                # Calculate the frame based error
                # groundTruthForError = np.array(outputTargetSampleFrames)[:,0:2]
                # totalCurrentError = totalCurrentError + FrameError(np.array(targetCarPoseList), np.array(targetCarGroundTruthPoseList))
                # totalLatError = totalLatError + FrameLatError(np.array(targetCarPoseList), np.array(targetCarGroundTruthPoseList))
                # totalLonError = totalLonError + FrameLonError(np.array(targetCarPoseList), np.array(targetCarGroundTruthPoseList))

                lonError = lonError + FrameLonError(np.array(targetCarPredictedPoses), np.array(targetCarGroundTruthPoseList))

                otherCarCount = 0
                for otherTrackedEligibleCarKey in trackerDict:

                    otherImageFiles = np.zeros((sample,temporal,OccupancyImageHeight,OccupancyImageWidth,channel))
                    othertargetCarImagePoseList = []

                    if len(trackerDict[otherTrackedEligibleCarKey]) > outputSize:
                        print('Tracker Data over populated!!!!!!!')
                        sys.exit()


                    if otherTrackedEligibleCarKey != primaryCarId and len(trackerDict[otherTrackedEligibleCarKey]) == outputSize:
                        otherCarCount = otherCarCount + 1
                        otherFolderPath = imageFolder + otherCarFolder + str(otherCarCount)
                        os.mkdir(otherFolderPath)
                        otherCarDataForPrediction = trackerDict[otherTrackedEligibleCarKey]

                        otherTargetLocalOriginX = otherCarDataForPrediction[temporal-1][0]
                        otherTargetLocalOriginY = otherCarDataForPrediction[temporal-1][1]

                        otherTargetLocalOriginXOriginal = otherCarDataForPrediction[temporal-1][0]
                        otherTargetLocalOriginYOriginal = otherCarDataForPrediction[temporal-1][1]

                        XFeetoffest = abs(otherTargetLocalOriginX - primaryLocalOriginXOriginal)
                        YFeetoffest = abs(otherTargetLocalOriginY - primaryLocalOriginYOriginal)

                        # if (XFeetoffest > observeWidth/2) or (YFeetoffest > observeLength):
                        #     continue

                        Xoffest = (otherTargetLocalOriginX - primaryLocalOriginXOriginal)*latResolution
                        Yoffest = (otherTargetLocalOriginY - primaryLocalOriginYOriginal)*lonResolution

                        for ndx,sampleOtherFrame in enumerate(otherCarDataForPrediction[0:inputSize]):
                            #concatinatedFrameArray = np.zeros((OccupancyImageHeight,OccupancyImageWidth,channel), dtype=np.float)
                            currentFrame = np.zeros((OccupancyImageHeight,OccupancyImageWidth), dtype=np.uint8)
                            currentFrame.fill(255)
                            #otherVelocityFrame = np.zeros((OccupancyImageHeight,OccupancyImageWidth), dtype=np.float)
                            #otherVelocityFrame.fill(0)
                            currentTargetX, currentTargetY, length, width, localTime, currentKey, otherSpeed = otherCarDataForPrediction[ndx]
                            relativeTargetX = imageOriginX + int((currentTargetX - otherTargetLocalOriginX)*latResolution)
                            relativeTargetY = imageOriginY - int((currentTargetY - otherTargetLocalOriginY)*lonResolution)
                            othertargetCarImagePoseList.append([relativeTargetX, relativeTargetY])
                            pixelLengthOtherTarget = int(length*lonResolution)
                            pixelWidthOtherTarget = int((width*latResolution)/2)
                            currentFrame[int(relativeTargetY):int(relativeTargetY+pixelLengthOtherTarget),int(relativeTargetX-pixelWidthOtherTarget):int(relativeTargetX+pixelWidthOtherTarget)] = 0
                            #otherVelocityFrame[int(relativeTargetY):int(relativeTargetY+pixelLength),int(relativeTargetX-pixelWidth):int(relativeTargetX+pixelWidth)] = otherSpeed
                            for forOtherTrackerKey in trackerDict:
                                corespndingOtherCarIndex = len(trackerDict[forOtherTrackerKey]) - outputSize + ndx
                                if corespndingOtherCarIndex >= 0 and forOtherTrackerKey!= currentKey and localTime == trackerDict[forOtherTrackerKey][corespndingOtherCarIndex][4]:
                                    currentOtherXCheck = trackerDict[forOtherTrackerKey][corespndingOtherCarIndex][0]
                                    currentOtherYCheck = trackerDict[forOtherTrackerKey][corespndingOtherCarIndex][1]
                                    # Ignore other cars those are out of the co ordinate frame
                                    absXDistance = abs(currentOtherXCheck-otherTargetLocalOriginX)
                                    absYDistance = abs(currentOtherYCheck-otherTargetLocalOriginY)
                                    if((absXDistance>=(occupancyMapWidth/2)) or (absYDistance>=(occupancyMapHeight/2))):
                                        continue

                                    absoluteXPixel = imageOriginX + int((currentOtherXCheck-otherTargetLocalOriginX)*latResolution)
                                    absoluteYPixel = imageOriginY - int((currentOtherYCheck-otherTargetLocalOriginY)*lonResolution)
                                    pixelLength = int(trackerDict[forOtherTrackerKey][corespndingOtherCarIndex][2]*lonResolution)
                                    pixelWidth = int((trackerDict[forOtherTrackerKey][corespndingOtherCarIndex][3]*latResolution)/2)
                                    currentFrame[int(absoluteYPixel):int(absoluteYPixel+pixelLength),int(absoluteXPixel-pixelWidth):int(absoluteXPixel+pixelWidth)] = 128
                                    #otherVelocityFrame[int(absoluteYPixel):int(absoluteYPixel+pixelLength),int(absoluteXPixel-pixelWidth):int(absoluteXPixel+pixelWidth)] = otherSpeed

                            # im = Image.fromarray(currentFrame)
                            # imageCount = imageCount+1
                            # imFileName = otherFolderPath + '/' + str(imageCount) + imageFileType
                            # im.save(imFileName)
                            otherImageFiles[0,ndx,:,:,0] = currentFrame/255
                            #otherImageFiles[0,ndx,:,:,1] = otherVelocityFrame/100

                        otherCarPredictedPose = []
                        
                        for t in range(1,predTemporal+1):
                            
                            predictedOtherPoses = model.predict(otherImageFiles)

                            # Remove the other folder after prediction
                            removeFolders = os.listdir(imageFolder)
                            for rmDir in removeFolders:
                                shutil.rmtree(imageFolder + rmDir)

                            # recreate the target forlders
                            os.mkdir(otherFolderPath)

                            # Collect the Next frame prediction
                            predictedPoseX =  predictedOtherPoses[0][0]
                            predictedPoseY =  predictedOtherPoses[0][1]

                            if predictedPoseY > OccupancyImageHeight/2:
                                print('In the wrong prediction')
                                predictedPoseY = (OccupancyImageHeight/2) - 1 ##- 2

                            
                            # Calculate the shift from the last position
                            shiftX = (OccupancyImageWidth/2) - predictedPoseX
                            shiftY = (OccupancyImageHeight/2) - predictedPoseY

                            if not otherCarPredictedPose:
                                otherCarPredictedPose.append([predictedPoseX,predictedPoseY])
                            else:
                                lastPredX = otherCarPredictedPose[-1][0]
                                lastPredY = otherCarPredictedPose[-1][1]
                                newPredX = lastPredX + shiftX #something weird check - shiftX
                                newPredY = lastPredY - shiftY
                                otherCarPredictedPose.append([newPredX,newPredY])

                            # Extract the first and second item to estimate the shift
                            firstX = othertargetCarImagePoseList[0][0]
                            firstY = othertargetCarImagePoseList[0][1]
                            secondX = othertargetCarImagePoseList[1][0]
                            secondY = othertargetCarImagePoseList[1][1]
                            newShiftX = firstX - secondX
                            newShiftY = firstY - secondY

                            othertargetCarImagePoseList = othertargetCarImagePoseList[1:]
                            othertargetCarImagePoseList.append([predictedPoseX,predictedPoseY])
                            # Update all the positions based on the new prediction 
                            for sdx, _ in enumerate(targetCarImagePoseList):
                                othertargetCarImagePoseList[sdx][0] = othertargetCarImagePoseList[sdx][0] + newShiftX
                                othertargetCarImagePoseList[sdx][1] = othertargetCarImagePoseList[sdx][1] + newShiftY
                            
                            # Update the origins
                            otherTargetLocalOriginX = otherTargetLocalOriginX + (shiftX*(1/latResolution))
                            otherTargetLocalOriginX = otherTargetLocalOriginX + (shiftY*(1/lonResolution))

                            otherImageFiles = np.zeros((sample,temporal,OccupancyImageHeight,OccupancyImageWidth,channel))

                            for qdx in range(0,temporal):
                                currentFrame = np.zeros((OccupancyImageHeight,OccupancyImageWidth), dtype=np.uint8)
                                currentFrame.fill(255)

                                # Draw the other target car
                                relativeTargetX = othertargetCarImagePoseList[qdx][0]
                                relativeTargetY = othertargetCarImagePoseList[qdx][1]
                                currentFrame[int(relativeTargetY):int(relativeTargetY+pixelLengthOtherTarget),int(relativeTargetX-pixelWidthOtherTarget):int(relativeTargetX+pixelWidthOtherTarget)] = 0

                                # Draw Other cars for other target car
                                for otherOthertrackerKey in trackerDict:
                                    # Extract the coresponding Index in the list for the other car
                                    shiftedTimeIndex = t + qdx
                                    localTime = otherCarDataForPrediction[shiftedTimeIndex][4]
                                    corespndingOtherCarIndex = len(trackerDict[otherOthertrackerKey]) - outputSize + shiftedTimeIndex
                                    if corespndingOtherCarIndex >= 0 and otherOthertrackerKey!= currentKey and localTime == trackerDict[otherOthertrackerKey][corespndingOtherCarIndex][4]:
                                        currentOtherXCheck = trackerDict[otherOthertrackerKey][corespndingOtherCarIndex][0]
                                        currentOtherYCheck = trackerDict[otherOthertrackerKey][corespndingOtherCarIndex][1]
                                        # Ignore other cars those are out of the co ordinate frame
                                        absXDistance = abs(currentOtherXCheck-otherTargetLocalOriginX)
                                        absYDistance = abs(currentOtherYCheck-otherTargetLocalOriginY)
                                        if((absXDistance>=(occupancyMapWidth/2)) or (absYDistance>=(occupancyMapHeight/2))):
                                            continue

                                        absoluteXPixel = imageOriginX + int((currentOtherXCheck-otherTargetLocalOriginX)*latResolution)
                                        absoluteYPixel = imageOriginY - int((currentOtherYCheck-otherTargetLocalOriginX)*lonResolution)
                                        pixelLength = int(trackerDict[otherOthertrackerKey][corespndingOtherCarIndex][2]*lonResolution)
                                        pixelWidth = int((trackerDict[otherOthertrackerKey][corespndingOtherCarIndex][3]*latResolution)/2)
                                        #currentFrame[int(absoluteYPixel),int(absoluteXPixel)] = 128
                                        currentFrame[int(absoluteYPixel):int(absoluteYPixel+pixelLength),int(absoluteXPixel-pixelWidth):int(absoluteXPixel+pixelWidth)] = 128

                                # im = Image.fromarray(currentFrame)
                                # imageCount = imageCount+1
                                # imFileName = otherFolderPath + '/' + str(imageCount) + imageFileType
                                primaryImageFiles[0,vdx,:,:,0] = currentFrame/255
                                #primaryImageFiles[0,ldx,:,:,1] = velocityFrame/100
                                # im.save(imFileName)

                        # Extract the ground truth
                        outputTargetSampleFrames = otherCarDataForPrediction[inputSize:outputSize]
                        otherTargetCarGroundTruthPoseList = []
                        for outputTargetSample in outputTargetSampleFrames:
                            currentTargetX, currentTargetY, length, width, localTime, unusedKey, unusedSpeed = outputTargetSample
                            relativeOutputTargetX = imageOriginX + int((currentTargetX - otherTargetLocalOriginXOriginal)*latResolution) + Xoffest
                            relativeOutputTargetY = imageOriginY - int((currentTargetY - otherTargetLocalOriginYOriginal)*lonResolution) - Yoffest
                            otherTargetCarGroundTruthPoseList.append([relativeOutputTargetX,relativeOutputTargetY])

                        # Plot the ground Truth
                        pts = np.array(otherTargetCarGroundTruthPoseList, np.int32)
                        pts = pts.reshape((-1,1,2))
                        cv2.polylines(opencvVisulaizeImage,[pts],False,(0,255,0),2)
                        textPose = pts[0,0]
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(opencvVisulaizeImage, str(int(otherTrackedEligibleCarKey)), (textPose[0],textPose[1]), font, 0.4,(0,0,255),1,cv2.LINE_AA)

                        # Update the predicted poses for other cars based on traget car using Xoffset and YOffset
                        for gdx,_ in enumerate(otherCarPredictedPose):
                            otherCarPredictedPose[gdx][0] = otherCarPredictedPose[gdx][0] + Xoffest
                            otherCarPredictedPose[gdx][1] = otherCarPredictedPose[gdx][1] - Yoffest

                        # Plot the Predicted 
                        pts = np.array(otherCarPredictedPose, np.int32)
                        pts = pts.reshape((-1,1,2))
                        cv2.polylines(opencvVisulaizeImage,[pts],False,(255,0,0),2)
                        textPose = pts[0,0]
                        font = cv2.FONT_HERSHEY_SIMPLEX





                rotated = np.rot90(opencvVisulaizeImage)
                rotated = np.rot90(rotated)
                rotated = np.rot90(rotated)

                cv2.imwrite(resultFolder + str(globalLonErrorCount) + '.png', rotated)


















                print('Predicted Poses \n')

                print(targetCarPredictedPoses)

                print('Ground Truth Poses \n')
                print(targetCarGroundTruthPoseList)

                print('Error count ' + str(globalLonErrorCount))

                print(lonError/globalLonErrorCount)

                
                # # Remove all the folders..
                removeFolders = os.listdir(imageFolder)
                for rmDir in removeFolders:
                    shutil.rmtree(imageFolder + rmDir)


                trackerDict[primaryCarId].pop(0)
                for trackerKey in trackerDict:
                    trackLength = len(trackerDict[trackerKey])
                    if trackLength >= outputSize and trackerKey!= primaryCarId:
                        trackerDict[trackerKey].pop(0)


if __name__ == '__main__':

    #PredictForAllCars(testTrajFilePath)
    PredictForTargetCars(testTrajFilePath)
    #PredictForAllCarsNew(testTrajFilePath)

    print('All the cars are prediect in the scene.')

    sys.exit()







