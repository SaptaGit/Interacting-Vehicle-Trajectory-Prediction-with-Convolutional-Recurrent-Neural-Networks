import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
 
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from PIL import Image
import numpy as np
from keras.models import Sequential
from keras.utils import Sequence
from keras.layers import LSTM, Dense, TimeDistributed, Conv2D, MaxPooling2D, Flatten, Dropout, MaxPooling1D
# import matplotlib.pyplot as plt
from keras import optimizers
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
import gc
import sys
import resource
import cv2
from keras import backend as K
from keras.callbacks import LearningRateScheduler
import math
from keras import callbacks

#K.set_image_dim_ordering('th')

imageWidth = 128
imageHeight = 1024
channel = 1
temporal = 30
trainFolder = '/home/saptarshi/PythonCode/AdvanceLSTM/data/train/'
valFolder = '/home/saptarshi/PythonCode/AdvanceLSTM/data/val/'
imageType = '.png'
outputFile = 'output.txt'
BatchSize = 1

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

    def __init__(self, dataList, batch_size, path):
        self.dataList = dataList
        self.batch_size = batch_size
        self.path = path


    def __len__(self):
        return int(np.ceil(len(self.dataList) / float(self.batch_size)))

    def __getitem__(self, idx):

        sampleInput = []
        sampleOutput = []
        #gather input grid map images
        currentBatchfileList = self.dataList[idx*self.batch_size:(idx + 1)*self.batch_size]
        #print(currentBatchfileList)
        for currentSample in currentBatchfileList:
            currentInputFiles = os.listdir(self.path + currentSample)
            currentInputImages = [i for i in currentInputFiles if i.endswith(imageType)]
            sortedImageFiles = sorted(currentInputImages, key=lambda a: int(a.split(".")[0]) )
            imageInput = []
            for ldx,imageFile in enumerate(sortedImageFiles):
                #imageData = (cv2.imread(path + currentSample + '/' +  imageFile, 0)/255).astype('float16')
                #print(imageData.shape)
                #imageInput.append(imageData)
                OGMMap = cv2.imread(self.path + currentSample + '/' +  imageFile, cv2.IMREAD_COLOR)
                positionInfo = (OGMMap[:,:,0]/255).astype('float16')
                speedInfo = (OGMMap[:,:,1]/100).astype('float16')
                imageInput.append(positionInfo)
                imageInput.append(speedInfo)
            sampleInput.append(imageInput)

            #gather output
            # extract the future position from the output.txt file as output for each sample
            f = open(self.path + currentSample + '/' +  outputFile,'r')
            outputLines = f.read().splitlines()
            f.close()
            outList = []
            for mdx,outputValue in enumerate(outputLines):
                outputX = float(outputValue.split(',')[0])
                outputY = float(outputValue.split(',')[1])
                outList.append([outputX,outputY])
            sampleOutput.append(outList[:temporal])     

        X = np.array(sampleInput).reshape(self.batch_size,temporal,imageHeight,imageWidth, channel)
        y = np.array(sampleOutput)

        return X,y 

def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.5
    epochs_drop = 40.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

class LossHistory(callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.lr = []
 
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.lr.append(step_decay(len(self.losses)))
        print('\n Current lr = ' + str(self.lr[-1]))

if __name__ == '__main__':

    print('Lets Start!!!! Sample count should be multiple of batch size')
    trainList = sorted(os.listdir(trainFolder), key=int)
    numberOfTrainSamples = len(trainList)
    print('Number of Train sample : ' + str(numberOfTrainSamples))

    valList = sorted(os.listdir(valFolder), key=int)
    numberOfValSamples = len(valList)
    print('Number of Val sample : ' + str(numberOfValSamples))

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

    loss_history = LossHistory()
    lrate = LearningRateScheduler(step_decay)
    callbacks_list = [loss_history, lrate]
    opt = Adam()

    model.compile(loss=euclidean_distance_loss, optimizer=opt, metrics=[euclidean_distance_loss])
    model.summary()

    trainGen = CIFAR10Sequence(trainList,BatchSize,trainFolder)
    valGen = CIFAR10Sequence(trainList,BatchSize,valFolder)
    stepsPerEpoch = numberOfTrainSamples // BatchSize
    #model.fit_generator(dataGen, steps_per_epoch=stepsPerEpoch, epochs=80, verbose=1)
    history = model.fit_generator(trainGen, steps_per_epoch=stepsPerEpoch, epochs=2, verbose=1, callbacks=callbacks_list, validation_data=valGen)
    model.save('TrainVal.h5')
    training_loss = history.history['loss']
    epoch_count = range(1, len(training_loss) + 1)
    # plt.plot(epoch_count, training_loss, 'r--')
    # plt.legend(['Training Loss'])
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.show()







