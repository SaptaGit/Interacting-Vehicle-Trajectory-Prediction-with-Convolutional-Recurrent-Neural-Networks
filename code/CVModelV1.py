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
testTrajFilePath = '/home/saptarshi/PythonCode/AdvanceLSTM/SplittedData/us101-0750am-0805am.csv' #i80-0400-0415.csv' 
# Load the pre trained model
modelFilePath = '/home/saptarshi/PythonCode/AdvanceLSTMServer/TrainedModels/ConcatExtraStateLeakyforLSTM.h5'
#model = load_model(modelFilePath, custom_objects={'EuclidianLoss': EuclidianLoss, 'EuclidianDistanceMetric' : EuclidianDistanceMetric})
# Set the Intermediate place holder image folder where it will save the image frames temporarily 
# before doing the prediction and clean the folder afterwards.
imageFolder = '/home/saptarshi/PythonCode/AdvanceLSTM/PlaceHolderForPrediction/'

cvImageFolder = '/home/saptarshi/PythonCode/AdvanceLSTM/cvImageFolder/'

# Remove any folders from the last if remaining from the place holder folder.
removeFolders = os.listdir(imageFolder)
for rmDir in removeFolders:
    shutil.rmtree(imageFolder + rmDir)

# Set the Folder for individual frame predicition results
resultFolder = '/home/saptarshi/PythonCode/AdvanceLSTM/Results/FrameResult/'
# Set the Folder for final global results (The predicted position plotted on the global scene)
globalResultFolder = '/home/saptarshi/PythonCode/AdvanceLSTM/Results/velocity/'
# This one is to pass some specific vehicle ids as list for situations like if lane change vehicles or turn vehicles.
#LaneChangeVehicleList = [54.0,115.0]  # Test with vehicle 11
####LaneChangeVehicleList = [121.0, 144.0]  # Test with vehicle 11
#####LaneChangeVehicleList = [159.0,142.0]  # for CV model and validation
# LaneChangeVehicleList = [159.0,169.0]  # for CV model and validation
#LaneChangeVehicleList = [74.0,84.0,66.0]  # god
#LaneChangeVehicleList = [121.0,144.0]  # for CV model and validation

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
#outputSize = 80
outputSize = 130
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
globalCVErrorCount = 0

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
# model = load_model(modelFilePath, custom_objects={'euclidean_distance_loss': euclidean_distance_loss, 'euclidean_distance_loss' : euclidean_distance_loss})

# print(model.get_weights())

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
    lonError = np.zeros(30)
    varError = []

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

    CVErrorArray = np.zeros(predTemporal)


        # Get the car ids to be processed
    totalVehicleList = []
    for vehicle in relativePositionList:
        currentVehicleID = vehicle[0][0][0]
        totalVehicleList.append(currentVehicleID)

    LaneChangeVehicleList = totalVehicleList[:150]  # test 150

    print('Vehicle IDs to be process are : ' + str(len(LaneChangeVehicleList)))



    #cv2.namedWindow('test', cv2.WINDOW_NORMAL)
    for vehicle in relativePositionList:
        currentVehicleID = vehicle[0][0][0]
        if(currentVehicleID not in LaneChangeVehicleList):
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
                #os.mkdir(imageFolder + targetCarFolder)
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

                inputPoses = targetCar[0:inputSize]
                lastTolastposeX = inputPoses[-2][0]
                lastTolastposeY = inputPoses[-2][1]

                lastposeX = inputPoses[-1][0]
                lastposeY = inputPoses[-1][1]

                shiftX = lastposeX - lastTolastposeX
                shiftY = lastposeY - lastTolastposeY

                # Predict using CV model
                CVError = []
                outputTargetSampleFrames = targetCar[inputSize:outputSize]
                global globalCVErrorCount
                globalCVErrorCount = globalCVErrorCount + 1
                cvGroundTruthList = []
                cvPredList = []

                for pdx in range(0,predTemporal):
                    groundTruthX = outputTargetSampleFrames[pdx][0]
                    groundTruthY = outputTargetSampleFrames[pdx][1]
                    cvPredPOseX = lastposeX + shiftX*(pdx+1)
                    cvPredPOseY = lastposeY + shiftY*(pdx+1)

                    diff = math.sqrt(((groundTruthX - cvPredPOseX)**2) + ((groundTruthY - cvPredPOseY)**2))
                    errorY = diff*feetToMeter
                    CVError.append(errorY)
                    cvGroundTruthList.append([groundTruthX,groundTruthY])
                    cvPredList.append([cvPredPOseX,cvPredPOseY])

                CVErrorArray = CVErrorArray + np.array(CVError)
                varError.append(CVError[:])
                print('Error for Vehicle : ' + str(currentVehicleID))
                print(CVErrorArray/globalCVErrorCount)

                initialX = cvGroundTruthList[0][0]
                initialY = cvGroundTruthList[0][1]

                for udx,eachGroundTruth in enumerate(cvGroundTruthList):
                    cvGroundTruthList[udx][0] = imageOriginX + (eachGroundTruth[0] - initialX)*latResolution
                    cvGroundTruthList[udx][1] = imageOriginY - (eachGroundTruth[1] - initialY)*lonResolution

                initialX = cvPredList[0][0]
                initialY = cvPredList[0][1]

                for udx,eachPred in enumerate(cvPredList):
                    cvPredList[udx][0] = imageOriginX + (eachPred[0] - initialX)*latResolution
                    cvPredList[udx][1] = imageOriginY - (eachPred[1] - initialY)*lonResolution

 
                currentFrame = np.zeros((OccupancyImageHeight,OccupancyImageWidth,3), dtype=np.uint8)
                currentFrame.fill(255)

                pts = np.array(cvGroundTruthList, np.int32)
                pts = pts.reshape((-1,1,2))
                cv2.polylines(currentFrame,[pts],False,(0,255,0),2)

                pts = np.array(cvPredList, np.int32)
                pts = pts.reshape((-1,1,2))
                cv2.polylines(currentFrame,[pts],False,(0,0,255),2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(currentFrame, str(primaryCarId), (50,200), font, 1,(0,0,255),1,cv2.LINE_AA)

                outputFileName = cvImageFolder + str(imageCount) + '.png'
                cv2.imwrite(outputFileName, currentFrame)
                imageCount = imageCount + 1

                trackerDict[primaryCarId].pop(0)
                for trackerKey in trackerDict:
                    trackLength = len(trackerDict[trackerKey])
                    if trackLength >= 60 and trackerKey!= primaryCarId:
                        trackerDict[trackerKey].pop(0)




    # with open('cverror.txt', 'w') as f:
    #     for item in varError:
    #         f.write("%s\n" % item)

    # f.close()
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


if __name__ == '__main__':

    #PredictForAllCars(testTrajFilePath)
    PredictForTargetCars(testTrajFilePath)

    print('All the cars are prediect in the scene.')

    sys.exit()







