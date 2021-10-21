import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
 
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="0"


from PIL import Image
import numpy as np
from keras.models import Sequential, Model
from keras.utils import Sequence
from keras.layers import LSTM, Dense, TimeDistributed, Conv2D, MaxPooling2D, Flatten, Dropout, MaxPooling1D, MaxoutDense, ConvLSTM2D, BatchNormalization, AveragePooling3D, LeakyReLU, Input, concatenate, Reshape, RepeatVector
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
channel = 2
temporal = 30
imageType = '.png'
outputFile = 'output.txt'
BatchSize = 256

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
    lonResolutionTFConstant = tf.constant(lonResolution)
    latResolutionTFConstant = tf.constant(latResolution)
    latLossEuclidian = tf.multiply(tf.square(y_true[:,:,0] - y_pred[:,:,0]), latResolutionTFConstant)
    lonLossEuclidian = tf.multiply(tf.square(y_true[:,:,1] - y_pred[:,:,1]), lonResolutionTFConstant)
    modifiedLatLoss = tf.multiply(latLossEuclidian,alpha)
    modifiedLonLoss = tf.multiply(lonLossEuclidian,(1-alpha))
    finalLoss = tf.sqrt(modifiedLatLoss + modifiedLonLoss)
    return finalLoss

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
        sampleClassLabel = []
        inputTrajectory = []
        #gather input grid map images
        currentBatchfileList = self.dataList[idx*self.batch_size:(idx + 1)*self.batch_size]
        for currentSample in currentBatchfileList:
            # currentInputFiles = os.listdir(self.path + currentSample)
            # currentInputImages = [i for i in currentInputFiles if i.endswith(imageType)]
            # sortedImageFiles = sorted(currentInputImages, key=lambda a: int(a.split(".")[0]) )
            # imageInput = []
            # for ldx,imageFile in enumerate(sortedImageFiles):
            #     OGMMap = cv2.imread(self.path + currentSample + '/' +  imageFile, cv2.IMREAD_COLOR)
            #     targetInfo = (OGMMap[:,:,0]/255).astype('float16')
            #     otherInfo = (OGMMap[:,:,1]/255).astype('float16')
            #     imageInput.append(targetInfo)
            #     imageInput.append(otherInfo)
            # sampleInput.append(imageInput)

            #gather output
            # extract the future position from the output.txt file as output for each sample
            f = open(self.path + currentSample + '/' +  outputFile,'r')
            outputLines = f.read().splitlines()
            f.close()
            inputTrajectoryStr = outputLines.pop().split(',')
            inputTrajectoryStr.pop() # removing the last item due to the last comma
            for kdx in range(0,len(inputTrajectoryStr),2):
                #inputTrajectory.append([inputTrajectoryStr[kdx], inputTrajectoryStr[kdx+1]])
                inputTrajectory.append([float(inputTrajectoryStr[kdx])/float(OccupancyImageWidth), float(inputTrajectoryStr[kdx+1])/float(OccupancyImageHeight)])

            # for inputTraj in inputTrajectoryStr:
            #     inputTrajectory.append(float(inputTraj))
            
            classLabelStr = outputLines.pop()
            tempClassVal = [float(classLabelStr.split(',')[0]), float(classLabelStr.split(',')[1]), float(classLabelStr.split(',')[2]), float(classLabelStr.split(',')[3]), float(classLabelStr.split(',')[4]), float(classLabelStr.split(',')[5])]
            sampleClassLabel.append(tempClassVal)
            outList = []
            for mdx,outputValue in enumerate(outputLines):
                outputX = float(outputValue.split(',')[0])
                outputY = float(outputValue.split(',')[1])
                outList.append([outputX,outputY])
            sampleOutput.append(outList)     

        #X = np.array(sampleInput).reshape(self.batch_size,temporal,OccupancyImageHeight,OccupancyImageWidth,channel)
        yClassLabel = np.array(sampleClassLabel)

        inputTrajArray = np.array(inputTrajectory).reshape(self.batch_size, temporal, 2)

        yTrajectory = np.array(sampleOutput).reshape(self.batch_size, temporal, 2)

        outTrajShifted = []
        for eachSample in sampleOutput:
            eachSampleShiftedOutTraj = []
            eachSampleShiftedOutTraj = [[OccupancyImageWidth/2,OccupancyImageHeight/2]] + eachSample[:-1]

            # for pdx,eachShifted in enumerate(eachSampleShiftedOutTraj):
            #     eachSampleShiftedOutTraj[pdx][0] = eachSampleShiftedOutTraj[pdx][0]/OccupancyImageWidth
            #     eachSampleShiftedOutTraj[pdx][1] = eachSampleShiftedOutTraj[pdx][1]/OccupancyImageHeight

            outTrajShifted.append(eachSampleShiftedOutTraj)

        outTrajShiftedArray = np.array(outTrajShifted).reshape(self.batch_size, temporal, 2)




        inputFinal = [inputTrajArray, outTrajShiftedArray]
        outputFinal = [yTrajectory]

        return inputFinal,outputFinal 

def step_decay(epoch):
    initial_lrate = 0.00001
    drop = 0.9
    epochs_drop = 45.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
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
    encoder1 = LSTM(n_units, return_sequences=True) 
    encoder = LSTM(n_units, return_state=True)
    encoder1_output = encoder1(encoder_inputs)
    encoder_outputs, state_h, state_c = encoder(encoder1_output)
    encoder_states = [state_h, state_c]
    # define training decoder
    decoder_inputs = Input(shape=(temporal,2))
    decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)

    #act = LeakyReLU(alpha=0.1)

    decoder_dense4 = Dense(256, activation='relu')
    decoder_outputs = decoder_dense4(decoder_outputs)

    decoder_dense3 = Dense(128, activation='relu')
    decoder_outputs = decoder_dense3(decoder_outputs)

    decoder_dense2 = Dense(64, activation='relu')
    decoder_outputs = decoder_dense2(decoder_outputs)
    
    decoder_dense1 = Dense(16, activation='relu')
    decoder_outputs = decoder_dense1(decoder_outputs)

    decoder_dense = Dense(2, activation='linear')
    decoder_outputs = decoder_dense(decoder_outputs)

    trainModel = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    # define inference encoder
    encoder_model = Model(encoder_inputs, encoder_states)
    # define inference decoder
    decoder_state_input_h = Input(shape=(n_units,))
    decoder_state_input_c = Input(shape=(n_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense4(decoder_outputs)
    decoder_outputs = decoder_dense3(decoder_outputs)
    decoder_outputs = decoder_dense2(decoder_outputs)
    decoder_outputs = decoder_dense1(decoder_outputs)
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    # trainModel = load_model('/home/saptarshi/PythonCode/AdvanceLSTM/trainmodelre.h5', custom_objects={'euclidean_distance_loss': euclidean_distance_loss})
    # encoder_model = load_model('/home/saptarshi/PythonCode/AdvanceLSTM/encoderre.h5')
    # decoder_model = load_model('/home/saptarshi/PythonCode/AdvanceLSTM/decoderre.h5')

    loss_history = LossHistory()
    lrate = LearningRateScheduler(step_decay)
    callbacks_list = [loss_history, lrate]
    opt = RMSprop()

    trainModel.compile(loss=euclidean_distance_loss, optimizer=opt)
    trainModel.summary()
    stepsPerEpoch = numberOfTrainSamples // BatchSize
    trainGen = CIFAR10Sequence(trainList,BatchSize,trainFolder)

    trainModel.fit_generator(trainGen, steps_per_epoch=stepsPerEpoch, epochs=1000, verbose=1, callbacks=callbacks_list)
    #trainModel.fit_generator(trainGen, steps_per_epoch=stepsPerEpoch, epochs=5000, verbose=1)

    trainModel.save('trainmodel.h5')
    encoder_model.save('encoder.h5')
    decoder_model.save('decoder.h5')


   