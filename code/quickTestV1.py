import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
 
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="0"


from PIL import Image
import numpy as np
from keras.models import Sequential, Model
from keras.utils import Sequence
from keras.layers import LSTM, Dense, GRU,TimeDistributed, Conv2D, MaxPooling2D, Flatten, Dropout, MaxPooling1D, MaxoutDense, ConvLSTM2D, BatchNormalization, AveragePooling3D, LeakyReLU, Input, concatenate, Reshape, RepeatVector, Bidirectional
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
trainFolder = '/home/saptarshi/PythonCode/AdvanceLSTM/SuperDuperhigh/'
#valFolder = '/home/saptarshi/PythonCode/AdvanceLSTM/BetterTrajVal/'

# Internal varaibles
# Set the different Occupancy Grid map and scene dimensions
occupancyMapWidth = 100
occupancyMapHeight = 600
OccupancyImageWidth = 1024
OccupancyImageHeight = 16384
lonResolution = OccupancyImageHeight/occupancyMapHeight 
latResolution = OccupancyImageWidth/occupancyMapWidth
lonResolutionSqr = lonResolution*lonResolution
latResolutionSqr = latResolution*latResolution
channel = 2
#temporal = 20
imageType = '.png'
outputFile = 'output.txt'
BatchSize = 128
inputTemporal = 20

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

        outputTrajecotry = [] 
        sampleClassLabel = [] 
        inputTrajectory = [] 
        decoderInputTraj = [] 

        currentBatchfileList = self.dataList[idx*self.batch_size:(idx + 1)*self.batch_size]
        for currentSample in currentBatchfileList:
            # Extract the Input trajectory 
            f = open(self.path + currentSample + '/' +  outputFile,'r')
            outputLines = f.read().splitlines()
            f.close()
            inputTrajectoryStr = outputLines.pop().split(',')
            inputTrajectoryStr.pop()  # removing the last item due to the last comma
            for kdx in range(20,len(inputTrajectoryStr),2):
                inputTrajectory.append([float(inputTrajectoryStr[kdx])/float(OccupancyImageWidth), float(inputTrajectoryStr[kdx+1])/float(OccupancyImageHeight)])

            # Extract the Ground truth label for Classification (Lane Change, Acc etc....)
            classLabelStr = outputLines.pop()
            tempClassVal = [float(classLabelStr.split(',')[0]), float(classLabelStr.split(',')[1]), float(classLabelStr.split(',')[2]), float(classLabelStr.split(',')[3]), float(classLabelStr.split(',')[4]), float(classLabelStr.split(',')[5])]
            sampleClassLabel.append(tempClassVal)

            # Extract the Gorunfd Truth position for the next frame
            outList = []
            outputX = float(outputLines[0].split(',')[0])
            outputY = float(outputLines[0].split(',')[1])
            outputTrajecotry.append([outputX,outputY])     

            # Prepeare the Decoder Input Trajecotry
            decoderInputTraj.append([OccupancyImageWidth/2,OccupancyImageHeight/2])


        # Convert Everything to array and reshape consdering the batch size        
        inputTrajArray = np.array(inputTrajectory).reshape(self.batch_size, inputTemporal, 2)  # Input Trajectpory
        decoderInputTraj = np.array(decoderInputTraj).reshape(self.batch_size, 1, 2)  # Decoder Input Trajectory (Shifted from the output)
        yClassLabel = np.array(sampleClassLabel)  # Class Label (Lane Change, Acc etc....)
        yTrajectory = np.array(outputTrajecotry).reshape(self.batch_size, 2)  # Output Trajectory

        # Collect the array needed for the model
        inputFinal = [inputTrajArray]
        outputFinal = [yTrajectory]

        return inputFinal,outputFinal 

def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.8
    epochs_drop = 5.0
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

# class EarlyStoppingByLossVal(callbacks.Callback):
#     def __init__(self, monitor='val_loss', value=1.4, verbose=0):
#         #super(Callback, self).__init__()
#         self.monitor = monitor
#         self.value = value
#         self.verbose = verbose

#     def on_epoch_end(self, epoch, logs={}):
#         current = logs.get(self.monitor)
#         if current is None:
#             warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

#         if current < self.value:
#             if self.verbose > 0:
#                 print("Epoch %05d: early stopping THR" % epoch)
#             self.model.stop_training = True


if __name__ == '__main__':

    trainList = sorted(os.listdir(trainFolder), key=int)
    numberOfTrainSamples = len(trainList)
    print('Number of Train sample : ' + str(numberOfTrainSamples))

    # valList = sorted(os.listdir(valFolder), key=int)
    # numberOfValSamples = len(valList)
    # print('Number of Validation sample : ' + str(numberOfValSamples))

    trajInput = Input(shape=(inputTemporal,2))
    alphaValue = 0.3
    regVal = 0.0001

    trajOutput, state_c1, state_h1 = LSTM(256, return_sequences=True, return_state=True)(trajInput)
    trajOutput = TimeDistributed(LeakyReLU(alpha=alphaValue))(trajOutput)
    trajOutput = TimeDistributed(BatchNormalization())(trajOutput)
    trajOutput, state_c2, state_h2  = LSTM(128, return_sequences=True, return_state=True)(trajOutput)
    trajOutput = TimeDistributed(LeakyReLU(alpha=alphaValue))(trajOutput)
    #trajOutput = TimeDistributed(BatchNormalization())(trajOutput)
    trajOutput = TimeDistributed(Dense(512))(trajOutput)
    trajOutput = TimeDistributed(LeakyReLU(alpha=alphaValue))(trajOutput)
    trajOutput = TimeDistributed(Dense(256))(trajOutput)
    trajOutput = TimeDistributed(LeakyReLU(alpha=alphaValue))(trajOutput)
    trajOutput = TimeDistributed(Dense(128))(trajOutput)
    trajOutput = TimeDistributed(LeakyReLU(alpha=alphaValue))(trajOutput)
    trajOutput = TimeDistributed(Dense(64))(trajOutput)
    trajOutput = TimeDistributed(LeakyReLU(alpha=alphaValue))(trajOutput)
    trajOutput = TimeDistributed(Dense(32))(trajOutput)
    trajOutput = TimeDistributed(LeakyReLU(alpha=alphaValue))(trajOutput)
    trajOutput = Flatten()(trajOutput)

    # stateConcatenated1 = concatenate([state_c1, state_h1])
    stateConcatenated2 = concatenate([state_c2, state_h2])

    # trajOutputFinal = concatenate([trajOutput, totalState])
    trajOutputFinal = concatenate([trajOutput, stateConcatenated2])

    #trajOutputFinal = BatchNormalization()(trajOutputFinal)

    trajOutputFinal = Dense(1024)(trajOutputFinal)
    trajOutputFinal = LeakyReLU(alpha=alphaValue)(trajOutputFinal)
    trajOutputFinal = Dense(512)(trajOutputFinal)
    trajOutputFinal = LeakyReLU(alpha=alphaValue)(trajOutputFinal)
    trajOutputFinal = Dense(256)(trajOutputFinal)
    trajOutputFinal = LeakyReLU(alpha=alphaValue)(trajOutputFinal)
    trajOutputFinal = Dense(128)(trajOutputFinal)
    trajOutputFinal = LeakyReLU(alpha=alphaValue)(trajOutputFinal)
    trajOutputFinal = Dense(64)(trajOutputFinal)
    trajOutputFinal = LeakyReLU(alpha=alphaValue)(trajOutputFinal)
    trajOutputFinal = Dense(32)(trajOutputFinal)
    trajOutputFinal = LeakyReLU(alpha=alphaValue)(trajOutputFinal)
    trajOutputFinal = Dense(16)(trajOutputFinal)
    trajOutputFinal = LeakyReLU(alpha=alphaValue)(trajOutputFinal)
    trajOutputFinal = Dense(2, activation='linear')(trajOutputFinal)

    loss_history = LossHistory()
    lrate = LearningRateScheduler(step_decay)
    callbacks_list = [loss_history, lrate]
    opt = RMSprop()

    model = Model(input=[trajInput], output=[trajOutputFinal])
    model.compile(loss=euclidean_distance_loss, optimizer=opt)
    model.summary()

    stepsPerEpoch = numberOfTrainSamples // BatchSize
    trainGen = CIFAR10Sequence(trainList,BatchSize,trainFolder)
    # valGen = CIFAR10Sequence(valList,BatchSize,valFolder)
    # history = model.fit_generator(trainGen, steps_per_epoch=stepsPerEpoch, epochs=200, verbose=1)
    history = model.fit_generator(trainGen, steps_per_epoch=stepsPerEpoch, epochs=300, verbose=1,  callbacks=callbacks_list)

    model.save('superDuperHighRes16384.h5')


 

   