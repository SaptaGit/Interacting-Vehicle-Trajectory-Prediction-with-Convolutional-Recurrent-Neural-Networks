import os
import PIL
from PIL import Image
import numpy as np
from keras.models import Sequential
from keras.utils import Sequence
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
import scipy.stats as stats
import math
import scipy.io as sio

#K.set_image_dim_ordering('th')

#imageVisualTest = Image.new('RGB', (512, 128))
#imageVisualTest = np.empty((512, 128))
imageVisualTest = np.zeros((1024,128,3), np.uint8)
occupancyMapWidth = 100
occupancyMapHeight = 600
OccupancyImageWidth = 128
OccupancyImageHeight = 512
imageOriginX = OccupancyImageWidth/2
imageOriginY = OccupancyImageHeight/2
lonResolution = OccupancyImageHeight/occupancyMapHeight
latResolution = OccupancyImageWidth/occupancyMapWidth
dataFolder = '/home/saptarshi/PythonCode/AdvanceLSTM/us-101TestData/'
BatchSize = 1
imageType = '.png'
outputFile = 'output.txt'
imageWidth = 128
imageHeight = 1024
channel = 1
temporal = 30
count = 0

groundTruth = []

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

def FrameError(y_true, y_pred):
    errorList = []
    for jdx,val in enumerate(y_true):
        diffX = pow(((y_true[jdx,0]-y_pred[jdx,0])*(1/latResolution)),2)
        diffY = pow(((y_true[jdx,1]-y_pred[jdx,1])*(1/lonResolution)),2)
        distError = math.sqrt(diffX+diffY)*0.3048
        errorList.append(distError)
    if (distError<0):
        print('error is : ' + str(distError))
    return np.array(errorList)

class CIFAR10Sequence(Sequence):

    def __init__(self, dataList, batch_size):
        self.dataList = dataList
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.dataList) / float(self.batch_size)))

    def __getitem__(self, idx):
        sampleInput = []
        sampleOutput = []
        #gather input grid map images
        currentBatchfileList = self.dataList[idx*self.batch_size:(idx + 1)*self.batch_size]
        for currentSample in currentBatchfileList:
            #print('processing Sample : ' + str(currentSample))
            currentInputFiles = os.listdir(dataFolder + currentSample)
            currentInputImages = [i for i in currentInputFiles if i.endswith(imageType)]
            sortedImageFiles = sorted(currentInputImages, key=lambda a: int(a.split(".")[0]) )
            imageInput = []
            for ldx,imageFile in enumerate(sortedImageFiles):
                imageData = (cv2.imread(dataFolder + currentSample + '/' +  imageFile, 0)/255).astype('float16')
                imageInput.append(imageData)
            sampleInput.append(imageInput)

            #gather output
            # extract the future position from the output.txt file as output for each sample
            f = open(dataFolder + currentSample + '/' +  outputFile,'r')
            outputLines = f.read().splitlines()
            f.close()
            outList = []
            for mdx,outputValue in enumerate(outputLines):
                outputX = float(outputValue.split(',')[0])
                outputY = float(outputValue.split(',')[1])
                outList.append([outputX,outputY])
            sampleOutput.append(outList) 
            if(len(outList) != 30):
                print('processing Sample : ' + str(currentSample))
                print(str(len(outList)))

        X = np.array(sampleInput).reshape(self.batch_size,temporal,imageHeight,imageWidth, channel)
        y = np.array(sampleOutput)
        groundTruth.append(y)

        return X,y 


if __name__ == '__main__':



    #model = load_model('TrainedModels/AdvanceLSTMV1.h5')
    model = load_model('/home/saptarshi/PythonCode/AdvanceLSTM/TrainedModels/415OGMapsFull20KSamples.h5', custom_objects={'euclidean_distance_loss': euclidean_distance_loss})
    #model.summary()
    #sys.exit()

    #trainSamples = LoadSamples()
    #numberOfSamples = len(trainSamples)
    #trainX,trainY = SplitTraininputOutput(trainSamples)
    #trainX = trainX.reshape(numberOfSamples,30,512,128,1)

    folderList = sorted(os.listdir(dataFolder), key=int)
    numberOfSamples = len(folderList)
    dataGen = CIFAR10Sequence(folderList,BatchSize)
    steps = np.ceil(numberOfSamples/BatchSize)
    batchOutput = model.predict_generator(dataGen, steps = steps)
    #groundTruthArray = np.zeros((numberOfSamples,30,2))

    #for ndx,_ in enumerate(groundTruth):
     #   groundTruthArray[ndx,:,:] = groundTruth[ndx][0,:,:]


    #groundTruthArray = np.array(groundTruth).reshape((numberOfSamples,temporal,2))

    FrameErrorArray = np.zeros(temporal)
    for ldx,output in enumerate(batchOutput):
        predicted = output
        #print(str(kdx))
        truePose = groundTruth[ldx].reshape(30,2)
        FrameErrorArray = FrameErrorArray + FrameError(truePose,predicted)

    FrameErrorArray = FrameErrorArray/numberOfSamples
    #sio.savemat('/home/saptarshi/PythonCode/AdvanceLSTM/MatlabScripts/FrameErrorUS101.mat', {'arr': FrameErrorArray})

    plt.plot(FrameErrorArray)
    plt.show()


    # For PositionError
    allPoseError = []
    for kdx,output in enumerate(batchOutput):
        predicted = output
        #print(str(kdx))
        truePose = groundTruth[kdx].reshape(30,2)
        allPoseError.append(PositionError(truePose,predicted))

    allPoseErrorArray = np.array(allPoseError)
    #sio.savemat('/home/saptarshi/PythonCode/AdvanceLSTM/MatlabScripts/ErrorUS101.mat', {'arr': allPoseErrorArray})
    meanError = np.mean(allPoseErrorArray)
    varError = np.std(allPoseErrorArray)

    print('Mean Error = ' + str(meanError) + ' Variance = ' + str(varError))

    # mu = meanError
    # variance = varError
    # sigma = math.sqrt(variance)
    # x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    # #plt.plot(x, stats.norm.pdf(x, mu, sigma))
    # plt.hist(allPoseErrorArray, bins=1000, range=[np.min(allPoseErrorArray), np.max(allPoseErrorArray)], edgecolor='r',linewidth=3)
    # plt.show()









