import os
from PIL import Image
import numpy as np
from keras.models import Sequential
from keras.utils import Sequence
from keras.layers import LSTM, Dense, TimeDistributed, Conv2D, MaxPooling2D, Flatten, Dropout, MaxPooling1D
import matplotlib.pyplot as plt
from keras import optimizers
from keras.utils import multi_gpu_model
import gc
import sys
import resource
import cv2
from keras import backend as K

#K.set_image_dim_ordering('th')

imageWidth = 128
imageHeight = 1024
channel = 2
temporal = 30
dataFolder = '/home/saptarshi/PythonCode/AdvanceLSTM/OccupancyMapsV2/'
imageType = '.png'
outputFile = 'output.txt'
BatchSize = 16

def euclidean_distance_loss(y_true, y_pred):
    """
    Euclidean distance loss
    https://en.wikipedia.org/wiki/Euclidean_distance
    :param y_true: TensorFlow/Theano tensor
    :param y_pred: TensorFlow/Theano tensor of the same shape as y_true
    :return: float
    """
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

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

        X = np.array(sampleInput).reshape(self.batch_size,temporal,imageHeight,imageWidth, channel)
        y = np.array(sampleOutput)

        return X,y 

if __name__ == '__main__':

    folderList = sorted(os.listdir(dataFolder), key=int)
    numberOfSamples = len(folderList)

    # Basic LSTM structure with Conv2D
    model = Sequential()
    # define CNN model
    model.add(TimeDistributed(Conv2D(4, (5,5), activation='relu', padding='same'), input_shape=(temporal,imageHeight,imageWidth,channel)))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(5,5))))
    model.add(TimeDistributed(Conv2D(8, (5,5), activation='relu', padding='same')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(5,5))))
    model.add(TimeDistributed(Conv2D(16, (5,5), activation='relu', padding='same')))
    model.add(TimeDistributed(Flatten()))
    #model.add(TimeDistributed(MaxPooling1D(pool_size=4)))
    # define LSTM model
    model.add(LSTM(256, activation='relu', return_sequences=True))
    model.add(LSTM(128, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(512, activation='relu')))
    #model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(256, activation='relu')))
    #model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(128, activation='relu')))
    model.add(TimeDistributed(Dense(64, activation='relu')))
    model.add(TimeDistributed(Dense(32, activation='relu')))
    model.add(TimeDistributed(Dense(2, activation='linear')))

    model.compile(loss=euclidean_distance_loss, optimizer='adam', metrics=[euclidean_distance_loss])
    model.summary()

    dataGen = CIFAR10Sequence(folderList,BatchSize)
    stepsPerEpoch = numberOfSamples // BatchSize
    model.fit_generator(dataGen, steps_per_epoch=stepsPerEpoch, epochs=20, verbose=1)
    #model.fit_generator(dataGen, steps_per_epoch=stepsPerEpoch, epochs=20, verbose=1, workers=10, use_multiprocessing=True)
    #model.save('AdvanceLSTMV8.h5')
    training_loss = history.history['loss']
    epoch_count = range(1, len(training_loss) + 1)
    plt.plot(epoch_count, training_loss, 'r--')
    plt.legend(['Training Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()







