import os
from PIL import Image
import numpy as np
from keras.models import Sequential
from keras.utils import Sequence
from keras.layers import LSTM, Dense, TimeDistributed, Conv2D, MaxPooling2D, Flatten, Dropout, MaxPooling1D, MaxoutDense
import matplotlib.pyplot as plt
from keras import optimizers
from keras.optimizers import RMSprop
from keras.utils import multi_gpu_model
import gc
import sys
import resource
import cv2
from keras import backend as K
import tensorflow as tf
from keras.callbacks import LearningRateScheduler
import math
from keras import callbacks

#K.set_image_dim_ordering('th')
# This is to pass the Occupancy Map Sample parent folder absolute path
dataFolder = '/home/saptarshi/PythonCode/AdvanceLSTM/foldertest/'

# Internal varaibles
# Set the different Occupancy Grid map and scene dimensions
occupancyMapWidth = 100
occupancyMapHeight = 600
OccupancyImageWidth = 128
OccupancyImageHeight = 1024
lonResolution = OccupancyImageHeight/occupancyMapHeight 
latResolution = OccupancyImageWidth/occupancyMapWidth
channel = 2
temporal = 30
imageType = '.png'
outputFile = 'output.txt'
BatchSize = 16

# # Custome Euclidian distance loss function (back up loss)
# def euclidean_distance_loss(y_true, y_pred):
#     #print(K.print_tensor(y_true, message='y_true = '))
#     #print("y_true = " + str(y_true.eval()))
#     #print('Inside loss!!!')
#     #d = y_true - y_pred
#     #d = tf.Print(d, [d], "Inside loss function..................................................")
#     #y_true=K.print_tensor(y_true)
#     #y_true = tf.Print(y_true, [y_true], "True:", summarize=100)
#     #y_pred = tf.Print(y_pred, [y_pred], "Predicted:", summarize=100)
#     #K.int_shape(y_true)
#     #return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))
#     alpha = tf.constant(0.5)
#     latLoss = (y_true[:,:,0] - y_pred[:,:,0])
#     latLoss = tf.Print(latLoss, [latLoss], 'latLoss:', summarize=100)
#     lonLoss = y_true[:,:,1] - y_pred[:,:,1]
#     modifiedLatLoss = tf.multiply(latLoss,(1-alpha))
#     #modifiedLonLoss = tf.multiply(latLoss,alpha)
#     modifiedLatLoss = tf.Print(modifiedLatLoss, [modifiedLatLoss], 'modifiedLatLoss**:', summarize=100)
#     loss2 = tf.reduce_mean(modifiedLatLoss)
#     #loss2 = y_true[:,1] - y_pred[:,1]
#     #loss3 = tf.reduce_sum(loss2 - loss1)
#     return loss2

# Custome Euclidian distance loss function
def CustomeLoss(y_true, y_pred):
    alpha = tf.constant(0.7)
    latLoss = tf.abs(y_true[:,:,0] - y_pred[:,:,0])
    #latLoss = tf.Print(latLoss, [latLoss], 'latLoss:', summarize=1000)
    lonLoss = tf.abs(y_true[:,:,1] - y_pred[:,:,1])
    modifiedLatLoss = tf.multiply(latLoss,alpha)
    modifiedLonLoss = tf.multiply(lonLoss,(1-alpha))
    finalLoss = modifiedLatLoss + modifiedLonLoss
    return finalLoss

# Orignial Euclidian Distance Error Metric
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

def step_decay(epoch):
    print('IN decay Function')
    initial_lrate = 0.001
    drop = 0.5
    epochs_drop = 5.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

class LossHistory(callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.lr = []
 
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.lr.append(step_decay(len(self.losses)))
        print('Current lr = ' + str(self.lr[-1]))

# Data Genarator Class
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
                OGMMap = cv2.imread(dataFolder + currentSample + '/' +  imageFile, cv2.IMREAD_GRAYSCALE)
                positionInfo = (OGMMap[:,:]/255).astype('float16')
                #speedInfo = (OGMMap[:,:,1]/100).astype('float16')
                imageInput.append(positionInfo)
                #imageInput.append(speedInfo)
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

        X = np.array(sampleInput).reshape(self.batch_size,temporal,OccupancyImageHeight*OccupancyImageWidth*channel)
        y = np.array(sampleOutput)

        return X,y 

if __name__ == '__main__':

    folderList = sorted(os.listdir(dataFolder), key=int)
    numberOfSamples = len(folderList)

    if(numberOfSamples%BatchSize != 0):
        print('Number of sample ' + str(numberOfSamples) + ' is not divided  by batch size ' + str(BatchSize))
        sys.exit()

    time_step = temporal
    data_dim = OccupancyImageHeight*OccupancyImageWidth*channel

    # Basic LSTM structure with Conv2D
    model = Sequential()
    # define CNN model
    #model.add(TimeDistributed(Conv2D(4, (5,5), activation='relu', padding='same'), input_shape=(temporal,OccupancyImageHeight,OccupancyImageWidth,channel)))
    #model.add(TimeDistributed(MaxPooling2D(pool_size=(5,5))))
    #model.add(TimeDistributed(Conv2D(8, (5,5), activation='relu', padding='same')))
    #model.add(TimeDistributed(MaxPooling2D(pool_size=(5,5))))
    #model.add(TimeDistributed(Conv2D(16, (5,5), activation='relu', padding='same')))
    #model.add(TimeDistributed(Flatten()))
    #model.add(TimeDistributed(MaxPooling1D(pool_size=4)))
    # define LSTM model
    model.add(LSTM(256, activation='relu', return_sequences=True, input_shape=(time_step,data_dim)))
    model.add(LSTM(128, activation='relu', return_sequences=True))
    model.add(TimeDistributed(MaxoutDense(512)))
    #model.add(Dropout(0.2))
    model.add(TimeDistributed(MaxoutDense(256)))
    #model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(128, activation='relu')))
    model.add(TimeDistributed(Dense(64, activation='relu')))
    model.add(TimeDistributed(Dense(32, activation='relu')))
    model.add(TimeDistributed(Dense(2, activation='linear')))

    loss_history = LossHistory()
    lrate = LearningRateScheduler(step_decay)
    callbacks_list = [loss_history, lrate]
    opt = RMSprop()

    model.compile(loss=EuclidianLoss, optimizer=opt, metrics=[EuclidianDistanceMetric])
    model.summary()

    dataGen = CIFAR10Sequence(folderList,BatchSize)
    stepsPerEpoch = numberOfSamples // BatchSize
    #history = model.fit_generator(dataGen, steps_per_epoch=stepsPerEpoch, epochs=40, verbose=1)
    history = model.fit_generator(dataGen, steps_per_epoch=stepsPerEpoch, epochs=40, verbose=1, callbacks=callbacks_list)
    #model.fit_generator(dataGen, steps_per_epoch=stepsPerEpoch, epochs=20, verbose=1, workers=10, use_multiprocessing=True)
    #model.save('EuclidianLossTemporal50.h5')
    training_loss = history.history['loss']
    epoch_count = range(1, len(training_loss) + 1)
    plt.plot(epoch_count, training_loss, 'r--')
    plt.legend(['Training Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()







