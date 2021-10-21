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
outputFile = 'output.txt'
BatchSize = 128
inputTemporal = 30

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

            # Extract the Gorunfd Truth position for the next 30 frame
            outList = []
            for r in range(0,30):
                outputX = float(outputLines[r].split(',')[0])
                outputY = float(outputLines[r].split(',')[1])
                outList.append([outputX,outputY])
            outputTrajecotry.append(outList)       

            # Prepeare the Decoder Input Trajecotry
            decodeShifted = [[OccupancyImageWidth/2, OccupancyImageHeight/2]] + outList[:-1]
            decoderInputTraj.append(decodeShifted)


        # Convert Everything to array and reshape consdering the batch size        
        inputTrajArray = np.array(inputTrajectory).reshape(self.batch_size, temporal, 2)  # Input Trajectpory
        decoderInputTraj = np.array(decoderInputTraj).reshape(self.batch_size, 1, 2)  # Decoder Input Trajectory (Shifted from the output)
        yClassLabel = np.array(sampleClassLabel)  # Class Label (Lane Change, Acc etc....)
        yTrajectory = np.array(outputTrajecotry).reshape(self.batch_size, temporal, 2)  # Output Trajectory

        # Collect the array needed for the model
        inputFinal = [inputTrajArray, decoderInputTraj]
        outputFinal = [yTrajectory]

        return inputFinal,outputFinal 

def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.7
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


if __name__ == '__main__':

    trainList = sorted(os.listdir(trainFolder), key=int)
    numberOfTrainSamples = len(trainList)
    print('Number of Train sample : ' + str(numberOfTrainSamples))
    n_units = 256

    # define training encoder
    encoder_inputs = Input(shape=(temporal,2))
    normLayer1 = BatchNormalization()
    normInput1 = normLayer1(encoder_inputs)
    encoder1 = LSTM(n_units, return_sequences=True, return_state=True)
    encoder_outputs, state_h1, state_c1 = encoder1(normInput1)
    normLayer2 = BatchNormalization()
    normInput2 = normLayer2(encoder_outputs)
    encoder2 = LSTM(n_units,  return_sequences=True, return_state=True)
    _ , state_h2, state_c2 = encoder2(normInput2)
    encoder_states = [state_h1, state_c1, state_h2, state_c2]

    # define training decoder
    decoder_inputs = Input(shape=(1,2))
    normLayer3 = BatchNormalization()
    decoder_norm1 = normLayer3(decoder_inputs)
    decoder_lstm1 = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_output1, _, _ = decoder_lstm1(decoder_norm1, initial_state=[state_h1, state_c1])
    normLayer4 = BatchNormalization()
    decoder_norm2 = normLayer4(decoder_output1)
    decoder_lstm2 = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_output2, _, _ = decoder_lstm2(decoder_norm2, initial_state=[state_h2, state_c2])
    #normLayer5 = BatchNormalization()
    #decoder_output2 = normLayer5(decoder_output2)

    leakyActivation = LeakyReLU()

    #decoder_dense10 = Dense(2048)
    #decoder_output2 = leakyActivation(decoder_dense10(decoder_output2))
    decoder_dense9 = Dense(1024)
    decoder_output2 = leakyActivation(decoder_dense9(decoder_output2))
    decoder_dense8 = Dense(512)
    decoder_output2 = leakyActivation(decoder_dense8(decoder_output2))
    decoder_dense7 = Dense(256)
    decoder_output2 = leakyActivation(decoder_dense7(decoder_output2))
    decoder_dense6 = Dense(128)
    decoder_output2 = leakyActivation(decoder_dense6(decoder_output2))
    decoder_dense5 = Dense(64)
    decoder_output2 = leakyActivation(decoder_dense5(decoder_output2))
    decoder_dense4 = Dense(32)
    decoder_output2 = leakyActivation(decoder_dense4(decoder_output2))
    decoder_dense3 = Dense(16)
    decoder_output2 = leakyActivation(decoder_dense3(decoder_output2))
    decoder_dense2 = Dense(8)
    decoder_output2 = leakyActivation(decoder_dense2(decoder_output2))
    decoder_dense1 = Dense(4)
    decoder_output2 = leakyActivation(decoder_dense1(decoder_output2))
    decoder_dense = Dense(2, activation='linear')
    decoder_output2 = decoder_dense(decoder_output2)

    trainModel = Model([encoder_inputs, decoder_inputs], decoder_output2)

    loss_history = LossHistory()
    lrate = LearningRateScheduler(step_decay)
    callbacks_list = [loss_history, lrate]
    opt = RMSprop()

    trainModel.compile(loss=euclidean_distance_loss, optimizer=opt)
    trainModel.summary()
    stepsPerEpoch = numberOfTrainSamples // BatchSize
    trainGen = CIFAR10Sequence(trainList,BatchSize,trainFolder)

    trainModel.fit_generator(trainGen, steps_per_epoch=stepsPerEpoch, epochs=1500, verbose=1, callbacks=callbacks_list)
    #trainModel.fit_generator(trainGen, steps_per_epoch=stepsPerEpoch, epochs=500, verbose=1)


    # define inference encoder
    encoder_model = Model(encoder_inputs, encoder_states)

    # define inference decoder
    decoder_state_input_h1 = Input(shape=(n_units,))
    decoder_state_input_c1 = Input(shape=(n_units,))
    decoder_state_input_h2 = Input(shape=(n_units,))
    decoder_state_input_c2 = Input(shape=(n_units,))
    decoder_states_inputs = [decoder_state_input_h1, decoder_state_input_c1, decoder_state_input_h2, decoder_state_input_c2]
    decoderNorm1 = normLayer3(decoder_inputs)
    decoder_outputs, deco_state_h1, deco_state_c1 = decoder_lstm1(decoderNorm1, initial_state=decoder_states_inputs[:2])
    decoder_outputs = normLayer4(decoder_outputs)
    decoder_outputs, deco_state_h2, deco_state_c2 = decoder_lstm2(decoder_outputs, initial_state=decoder_states_inputs[-2:])
    decoder_states = [deco_state_h1, deco_state_c1, deco_state_h2, deco_state_c2]
    decoder_outputs = leakyActivation(decoder_dense9(decoder_outputs))
    decoder_outputs = leakyActivation(decoder_dense8(decoder_outputs))
    decoder_outputs = leakyActivation(decoder_dense7(decoder_outputs))
    decoder_outputs = leakyActivation(decoder_dense6(decoder_outputs))
    decoder_outputs = leakyActivation(decoder_dense5(decoder_outputs))
    decoder_outputs = leakyActivation(decoder_dense4(decoder_outputs))
    decoder_outputs = leakyActivation(decoder_dense3(decoder_outputs))
    decoder_outputs = leakyActivation(decoder_dense2(decoder_outputs))
    decoder_outputs = leakyActivation(decoder_dense1(decoder_outputs))
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    trainModel.save('./trainmodel.h5')
    encoder_model.save('./encoder.h5')
    decoder_model.save('./decoder.h5')

   