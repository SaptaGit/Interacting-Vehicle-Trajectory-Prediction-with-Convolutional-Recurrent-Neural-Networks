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

#K.set_image_dim_ordering('th')

#imageVisualTest = Image.new('RGB', (512, 128))
#imageVisualTest = np.empty((512, 128))
imageVisualTest = np.zeros((512,128,3), np.uint8)

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
    folderPath = '/home/saptarshi/PythonCode/AdvanceLSTM/TestData/layerTest/'
    folderList = []
    for i,j,y in sorted(os.walk(folderPath)):
        folderList.append(i)

    # Pop out the first item as this is holding the parent folder
    folderList.pop(0)

    allSample = []
    imageType = '.png'
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

def ReturnLayerImage(layerNumber, model, data):
    imageSavePath = '/home/saptarshi/PythonCode/AdvanceLSTM/IntermediateImages/'
    check = model.layers[layerNumber].output
    intermediate_layer_model = Model(inputs=model.input, outputs=check)
    intermediate_output = intermediate_layer_model.predict(data)
    filterCount = intermediate_output.shape[-1]
    intensity = intermediate_output[0,0,:,:,:].sum(axis=2)
    imageName = 'Layer_' + str(layerNumber) + '.png'
    cv2.imwrite(imageSavePath+imageName,intensity*255)


if __name__ == '__main__':

    #model = load_model('TrainedModels/AdvanceLSTMV1.h5')
    model = load_model('/home/saptarshi/PythonCode/AdvanceLSTM/TrainedModels/415OGMapsFull20KSamples.h5', custom_objects={'euclidean_distance_loss': euclidean_distance_loss})

    trainSamples = LoadSamples()
    numberOfSamples = len(trainSamples)
    trainX,trainY = SplitTraininputOutput(trainSamples)
    trainX = trainX.reshape(numberOfSamples,30,512,128,1)
    predicted = model.predict(trainX)
    
    ReturnLayerImage(0,model,trainX)
    for val in range(5):
        ReturnLayerImage(val,model,trainX)

    cv2.namedWindow('intermediate', cv2.WINDOW_NORMAL)
    intensity = intermediatImage.sum(axis=2)
    cv2.imshow('intermediate', intensity*255)
    cv2.waitKey(0)
    groundTruth = np.reshape(trainY,(30,2))
    predictedReshaped = np.reshape(predicted,(30,2))

    if(len(groundTruth) != len(predictedReshaped)):
        print('Mismatched shape for ground truth and Predicted')
        sys.exit()

    predictionLength = len(predictedReshaped)
    
    cv2.namedWindow('test', cv2.WINDOW_NORMAL)

    global imageVisualTest
    colorcv = cv2.cvtColor(imageVisualTest,cv2.COLOR_GRAY2RGB)
    # resizedIMage = imageVisualTest.resize((256, 1024), PIL.Image.ANTIALIAS)
    # opencvImage = np.array(resizedIMage) 

    #cv2.line(imageVisualTest,(0,0),(511,511),(255,0,0),1)
    #cv2.imshow('test', imageVisualTest)
    #cv2.waitKey(0)

    groundTruthPosXLast = int(groundTruth[0][0])
    groundTruthPosYLast = int(groundTruth[0][1])
    predictedPosXLast = int(predictedReshaped[0][0])
    predictedPosYLast = int(predictedReshaped[0][1])

    for idx in range(1,predictionLength):
        groundTruthPosX = int(groundTruth[idx][0])
        groundTruthPosY = int(groundTruth[idx][1])
        predictedPosX = int(predictedReshaped[idx][0])
        predictedPosY = int(predictedReshaped[idx][1])
        cv2.line(colorcv, (groundTruthPosXLast,groundTruthPosYLast), (groundTruthPosX,groundTruthPosY), (255,0,0), 1)
        cv2.line(colorcv,(predictedPosXLast,predictedPosYLast), (predictedPosX,predictedPosY), (0,0,255), 1)
        groundTruthPosXLast = groundTruthPosX
        groundTruthPosYLast = groundTruthPosY
        predictedPosXLast = predictedPosX
        predictedPosYLast = predictedPosY

    pilImage = Image.fromarray(colorcv)
    resizedPILIMage = pilImage.resize((512, 2048), PIL.Image.ANTIALIAS)
    opencvImage = np.array(resizedPILIMage)
    #resizedIMage = imageVisualTest.resize((256, 1024), PIL.Image.ANTIALIAS)
    #resized = cv2.resize(imageVisualTest, (512, 2048), interpolation = cv2.INTER_CUBIC)
    #opencvImage = np.array(resizedIMage) 
    cv2.imwrite('test.png', opencvImage)
    cv2.waitKey(0)

    print('done....')








