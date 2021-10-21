import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
 
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="1"


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
# This is to pass the test folder path parent
testFolder = '/home/saptarshi/PythonCode/AdvanceLSTM/PositionDotTestData/'
valFolder = '/media/disk1/sap/AdvanceLSTMServer/PositionDotValidateData/'

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
BatchSize = 256
globalLonErrorCount = 0
feetToMeter = 0.3048

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

# Calculate the average longitudinal position error corresponding to the future frame count
def FrameLonError(y_true, y_pred):
    errorList = []
    for jdx,val in enumerate(y_true):
        diffY = abs((y_true[jdx,1]-y_pred[jdx,1])*(1/lonResolution)*feetToMeter)
        errorList.append(diffY)
    if (diffY<0):
        print('error is : ' + str(diffY))
    global globalLonErrorCount
    globalLonErrorCount = globalLonErrorCount + 1
    print('Global Error Count = ' + str(globalLonErrorCount))
    return np.array(errorList)

def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.5
    epochs_drop = 10.0
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

    trainModel = load_model('/home/saptarshi/PythonCode/AdvanceLSTMServer/TrainedModels/trainmodel.h5', custom_objects={'EuclidianLoss': EuclidianLoss})
    encoder_model = load_model('/home/saptarshi/PythonCode/AdvanceLSTMServer/TrainedModels/encoder.h5')
    decoder_model = load_model('/home/saptarshi/PythonCode/AdvanceLSTMServer/TrainedModels/decoder.h5')

    testFolderList = os.listdir(testFolder)
    modifiedLonError = np.zeros((30))

    #read input data for all the folders in the test folder

    for eachFolder in testFolderList:
        filePath = testFolder + eachFolder + '/output.txt'
        f = open(filePath, 'r')

        # Read input Trajecotry 
        outputLines = f.read().splitlines()
        inputTestTrajstr = outputLines[-1].rstrip(',')
        inputTestTrajList = inputTestTrajstr.split(',')
        inputTestTraj = []
        for kdx in range(20,len(inputTestTrajList),2):
            testX = float(inputTestTrajList[kdx])/float(OccupancyImageWidth)
            testY = float(inputTestTrajList[kdx+1])/float(OccupancyImageHeight)
            inputTestTraj.append([testX,testY])
        inputTestArray = np.array(inputTestTraj).reshape(1,temporal,2)

        # Read ground truth traj
        outputTrajStr = outputLines[0:30]
        outputTrajList = []
        for eachOutStr in outputTrajStr:
            outputX = float(eachOutStr.split(',')[0])
            outputY = float(eachOutStr.split(',')[1])
            outputTrajList.append([outputX,outputY])
        outputTrajArray = np.array(outputTrajList)


        # Originial encode
        # Original    encode
        # state = encoder_model.predict(inputTestArray)
        # # start of sequence input
        # targetSeq = []
        # targetSeq.append([OccupancyImageWidth/2,OccupancyImageHeight/2])
        # for ndx in range(0,29):
        #     #targetSeq.append([OccupancyImageWidth/2,OccupancyImageHeight/2])
        #     targetSeq.append([0,0])


        # target_seq = np.array(targetSeq).reshape(1,temporal,2)
        # # collect predictions
        # outputOriginal = list()
        # for t in range(30):
        #     # predict next char
        #     yhat, h, c, h1, c1 = decoder_model.predict([target_seq] + state)
        #     # store prediction
        #     outputOriginal.append(yhat[0,0])
        #     # update state
        #     state = [h, c, h1, c1]
        #     # update target sequence
        #     target_seq[0,t,:] = yhat[0,0,:]
        #     #target_seq = yhat
        
        # resultOriginal = np.array(outputOriginal)
        
        
        # Modified encode
        state = encoder_model.predict(inputTestArray)
        # start of sequence input
        targetSeq = []
        #for ndx in range(0,30):
        for ndx in range(0,1):
            targetSeq.append([OccupancyImageWidth/2,OccupancyImageHeight/2])

        target_seq = np.array(targetSeq).reshape(1,1,2)

        # collect predictions
        outputModified = list()
        newInput = np.zeros((1,30,2))
        for t in range(30):
            # predict next pose
            yhat1 = decoder_model.predict([target_seq] + state)
            # store prediction
            lastX = inputTestArray[:,-1,0]*OccupancyImageWidth
            lastY = inputTestArray[:,-1,1]*OccupancyImageHeight
            predX = yhat1[0][0][0][0]
            predY = yhat1[0][0][0][1]
            if predY > 1024:
                predY = 1023

            # if (t==0):
            #     predX = yhat1[0][0][0][0]
            #     predY = yhat1[0][0][0][1]
            # else:
            #     predX = (yhat1[0][0][0][0]*0.5) + 0.5*lastPredXFilter
            #     predY = (yhat1[0][0][0][1]*0.5) + 0.5*lastPredYFilter
            
            #lastPredXFilter = predX
            #lastPredYFilter = predY

            shiftX = lastX - predX
            shiftY = lastY - predY

            firstX = inputTestArray[0,0,0]*OccupancyImageWidth
            firstY = inputTestArray[0,0,1]*OccupancyImageHeight

            secondX = inputTestArray[0,1,0]*OccupancyImageWidth
            secondY = inputTestArray[0,1,1]*OccupancyImageHeight

            newShiftX = firstX - secondX
            newShiftY = firstY - secondY

            inputTestArray[:,:-1,:] = inputTestArray[:,1:,:]
            inputTestArray[:,-1,:] = np.array([predX/OccupancyImageWidth, predY/OccupancyImageHeight])
            for pdx in range(0,int(temporal)):
                inputTestArray[0,pdx,0] = ((inputTestArray[0,pdx,0]*OccupancyImageWidth) + newShiftX)/OccupancyImageWidth
                inputTestArray[0,pdx,1] = ((inputTestArray[0,pdx,1]*OccupancyImageHeight) + newShiftY)/OccupancyImageHeight

            if not outputModified:
                outputModified.append(np.array([yhat1[0][0][0][0], yhat1[0][0][0][1]]))
            else:
                lastPredX = outputModified[-1][0]
                lastPredY = outputModified[-1][1] 
                currentPredX = yhat1[0][0][0][0]
                currentPredY = yhat1[0][0][0][1]
                predShiftX = (OccupancyImageWidth/2) - currentPredX
                predShiftY = (OccupancyImageHeight/2) - currentPredY
                newPredX = yhat1[0][0][0][0] #lastPredX - predShiftX
                newPredY = lastPredY - predShiftY
                outputModified.append(np.array([newPredX,newPredY]))

            state = encoder_model.predict(inputTestArray)
    
        resultModified = np.array(outputModified)

        # fileWrite = open(testFolder + eachFolder + '/outputNew.txt', 'a')
        # for outVal in outputModified:
        #     outX = outVal[0]
        #     outY = outVal[1]
        #     fileWrite.write(str(outX) + ',' + str(outY) + '\n')

        # for inVal in outputTrajList:
        #     outX = inVal[0]
        #     outY = inVal[1]
        #     fileWrite.write(str(outX) + ',' + str(outY) + '\n')

        # fileWrite.close()



        modifiedLonError = modifiedLonError + FrameLonError(outputTrajArray,resultModified)
        print(modifiedLonError/globalLonErrorCount)


    #plt.plot(originalLonError, label='Original')
    plt.plot(modifiedLonError/globalLonErrorCount, label='Modified')
    plt.legend()
    plt.show()




    # # Modified with 5 step
    # # encode
    # state = encoder_model.predict(inputTestArray)
    # # start of sequence input
    # targetSeq = []
    # for ndx in range(0,30):
    #     targetSeq.append([OccupancyImageWidth/2,OccupancyImageHeight/2])
    #     #targetSeq.append([0,0])


    # target_seq = np.array(targetSeq).reshape(1,temporal,2)
    # # collect predictions
    # output = list()

    # for k in range(6):
    #     state = encoder_model.predict(inputTestArray)
    #     for t in range(5):
    #         # predict next char
    #         yhat, h, c = decoder_model.predict([target_seq] + state)

    #         # store prediction
    #         output.append(yhat[0,0,:])

    #         # update state
    #         state = [h, c]

    #         # update target sequence
    #         target_seq = yhat
        
    #     inputTestArray[:,:-5,:] = inputTestArray[:,5:,:]
    #     inputTestArray[:,-5:,0] = (np.array(output[-5:])).reshape((1,5,1))
    #     for pdx in range(0,30):
    #         inputTestArray[0,pdx,0] = ((inputTestArray[0,pdx,0]*OccupancyImageWidth) - shiftX)/OccupancyImageWidth
    #         inputTestArray[0,pdx,1] = ((inputTestArray[0,pdx,1]*OccupancyImageHeight) - shiftY)/OccupancyImageHeight
    
    # result = np.array(output)

    # print(result)







