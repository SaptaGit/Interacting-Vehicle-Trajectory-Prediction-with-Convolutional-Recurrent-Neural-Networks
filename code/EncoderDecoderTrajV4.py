import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
 
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="0"


from PIL import Image
import numpy as np
from keras.models import Sequential, Model
from keras.utils import Sequence
from keras.layers import LSTM, Dense, GRU,TimeDistributed, Conv2D, MaxPooling2D, Flatten, Dropout, MaxPooling1D, MaxoutDense, ConvLSTM2D, BatchNormalization, AveragePooling3D, LeakyReLU, Input, concatenate, Reshape, RepeatVector
import matplotlib.pyplot as plt
from keras import optimizers
from keras.optimizers import RMSprop,Adam
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
from keras.utils import plot_model
from keras.models import load_model
from keras.layers.advanced_activations import LeakyReLU


#K.set_image_dim_ordering('th')
# This is to pass the Occupancy Map Sample parent folder absolute path
trainFolder = '/home/saptarshi/PythonCode/AdvanceLSTM/Vehicle11/'
#valFolder = '/media/disk1/sap/AdvanceLSTMServer/PositionDotValidateData/'

# Internal varaibles
# Set the different Occupancy Grid map and scene dimensions
occupancyMapWidth = 100
occupancyMapHeight = 600
OccupancyImageWidth = 128
OccupancyImageHeight = 1024
lonResolution = OccupancyImageHeight/occupancyMapHeight 
latResolution = OccupancyImageWidth/occupancyMapWidth
lonResolutionSqr = lonResolution*lonResolution
latResolutionSqr = latResolution*latResolution
channel = 2
temporal = 30
imageType = '.png'
outputFile = 'outputNew.txt'
BatchSize = 128
inputTemporal = 30
#checkModel = load_model('/home/saptarshi/PythonCode/AdvanceLSTM/encoder.h5')

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

def euclidean_distance_loss(y_true, y_pred):
    """
    Euclidean distance loss
    https://en.wikipedia.org/wiki/Euclidean_distance
    :param y_true: TensorFlow/Theano tensor
    :param y_pred: TensorFlow/Theano tensor of the same shape as y_true
    :return: float
    """
    #y_true = tf.Print(y_true, [y_true], "True:", summarize=100)
    #y_pred = tf.Print(y_pred, [y_pred], "Predicted:", summarize=100)
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

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
    lonResolutionTFConstant = tf.constant(lonResolutionSqr)
    latResolutionTFConstant = tf.constant(latResolutionSqr)
    latLossEuclidian = tf.multiply(tf.square(y_true[:,:,0] - y_pred[:,:,0]), latResolutionTFConstant)
    lonLossEuclidian = tf.multiply(tf.square(y_true[:,:,1] - y_pred[:,:,1]), lonResolutionTFConstant)
    #modifiedLatLoss = tf.multiply(latLossEuclidian,alpha)
    #modifiedLonLoss = tf.multiply(lonLossEuclidian,(1-alpha))
    finalLoss = tf.sqrt(latLossEuclidian + lonLossEuclidian)
    return finalLoss

class CIFAR10Sequence(Sequence):

    def __init__(self, dataList, batch_size, path):
        self.dataList = dataList
        self.batch_size = batch_size
        self.path = path

    def __len__(self):
        return int(np.ceil(len(self.dataList) / float(self.batch_size)))

    def __getitem__(self, idx):

        predictedTrajecotry = [] 
        groundTruthTrajecotry = [] 

        currentBatchfileList = self.dataList[idx*self.batch_size:(idx + 1)*self.batch_size]
        for currentSample in currentBatchfileList:

            f = open(self.path + currentSample + '/' +  outputFile,'r')
            outputLines = f.read().splitlines()
            f.close()

            # Extract the Predicted position for the next 30 frame
            predList = []
            for r in range(0,30):
                outputX = float(outputLines[r].split(',')[0])/float(OccupancyImageWidth)
                outputY = float(outputLines[r].split(',')[1])/float(OccupancyImageHeight)
                predList.append([outputX,outputY])
            predictedTrajecotry.append(predList)   

            # Extract the Ground Truth position for the next 30 frame
            truthList = []
            for j in range(30,60):
                outputX = float(outputLines[j].split(',')[0])
                outputY = float(outputLines[j].split(',')[1])
                truthList.append([outputX,outputY])
            groundTruthTrajecotry.append(truthList)     

        # Convert Everything to array and reshape consdering the batch size        
        predictedTrajecotryArray = np.array(predictedTrajecotry).reshape(self.batch_size, temporal, 2)
        groundTruthTrajecotryArray = np.array(groundTruthTrajecotry).reshape(self.batch_size, temporal, 2)  # Decoder Input Trajectory (Shifted from the output)


        # Collect the array needed for the model
        inputFinal = [predictedTrajecotryArray]
        outputFinal = [groundTruthTrajecotryArray]

        return inputFinal,outputFinal 

def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.7
    epochs_drop = 20.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    if lrate < 0.0000001:
        lrate = 0.0000001
    return lrate

class LossHistory(callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.losses = []
        #self.vallosses = []
        self.lr = []
 
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        #self.vallosses.append(logs.get('val_loss'))
        self.lr.append(step_decay(len(self.losses)))
        #print('\n Current lr = ' + str(self.lr[-1]))
        #print('\n Current Val Loss = ' + str(self.vallosses[-1]))


if __name__ == '__main__':

    trainList = sorted(os.listdir(trainFolder), key=int)
    numberOfTrainSamples = len(trainList)
    print('Number of Train sample : ' + str(numberOfTrainSamples))
    n_units = 256

    checkInput = Input(shape=(30,2))
    noiseModel = Sequential()
    noiseModel.add(LSTM(n_units, return_sequences=True, input_shape=(temporal,2)))
    noiseModel.add(TimeDistributed(BatchNormalization()))
    
    noiseModel.add(LSTM(n_units, return_sequences=True))
    noiseModel.add(TimeDistributed(BatchNormalization()))
    noiseModel.add(TimeDistributed(Dense(128)))
    noiseModel.add(TimeDistributed(LeakyReLU()))
    noiseModel.add(TimeDistributed(Dense(64)))
    noiseModel.add(TimeDistributed(LeakyReLU()))
    noiseModel.add(TimeDistributed(Dense(32)))
    noiseModel.add(TimeDistributed(LeakyReLU()))
    noiseModel.add(TimeDistributed(Dense(16)))
    noiseModel.add(TimeDistributed(LeakyReLU()))
    noiseModel.add(TimeDistributed(Dense(2, activation='linear')))

    loss_history = LossHistory()
    lrate = LearningRateScheduler(step_decay)
    callbacks_list = [loss_history, lrate]
    opt = RMSprop()

    noiseModel.compile(loss=euclidean_distance_loss, optimizer=opt)
    noiseModel.summary()
    stepsPerEpoch = numberOfTrainSamples // BatchSize
    trainGen = CIFAR10Sequence(trainList,BatchSize,trainFolder)
    noiseModel.fit_generator(trainGen, steps_per_epoch=stepsPerEpoch, epochs=500, verbose=1, callbacks=callbacks_list)

    noiseModel.save('denoise.h5')




   