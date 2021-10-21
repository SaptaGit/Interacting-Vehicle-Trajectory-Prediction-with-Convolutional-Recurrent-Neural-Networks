import os
from PIL import Image
import numpy as np
from keras.models import Sequential
from keras.utils import Sequence
from keras.layers import LSTM, Dense, TimeDistributed, Conv2D, MaxPooling2D, Flatten
import matplotlib.pyplot as plt
import sys
import cv2
import gc
#from keras import backend as K

#K.set_image_dim_ordering('th')

imageWidth = 128
imageHeight = 1024
channel = 1
temporal = 30
dataFolder = '/home/saptarshi/PythonCode/AdvanceLSTM/OccupancyMapsV2/'
imageType = '.png'
outputFile = 'output.txt'

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
        #gather input
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

    #trainX, trainY = LoadSamples()
    folderList = sorted(os.listdir(dataFolder), key=int)
    numberOfSamples = len(os.listdir(dataFolder))

    # Basic LSTM structure with Conv2D
    model = Sequential()
    # define CNN model
    model.add(TimeDistributed(Conv2D(32, (3,3), activation='relu', padding='same'), batch_input_shape=(numberOfSamples,30,512,128,1)))
    model.add(TimeDistributed(Conv2D(64, (3,3), activation='relu', padding='same')))
    model.add(TimeDistributed(Conv2D(128, (3,3), activation='relu', padding='same')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    model.add(TimeDistributed(Flatten()))
    # define LSTM model
    model.add(LSTM(256, activation='relu', return_sequences=True))
    model.add(LSTM(128, activation='relu', return_sequences=True))
    model.add(Dense(2, activation='relu'))
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    #history = model.fit(trainX, trainY, epochs=8, verbose=1)
    batch_size = 16
    gen = CIFAR10Sequence(folderList,2)
    model.fit_generator(gen, steps_per_epoch=len(os.listdir(dataFolder)) // batch_size, epochs=20, verbose=1, workers=10, use_multiprocessing=True)

    model.save('AdvanceLSTM.h5')
    training_loss = history.history['loss']
    epoch_count = range(1, len(training_loss) + 1)
    plt.plot(epoch_count, training_loss, 'r--')
    plt.legend(['Training Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()







