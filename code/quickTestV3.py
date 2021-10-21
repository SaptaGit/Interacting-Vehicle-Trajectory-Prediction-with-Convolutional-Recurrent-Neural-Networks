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
trainFolder = '/home/saptarshi/PythonCode/AdvanceLSTM/quickTestForBad/'
#valFolder = '/home/saptarshi/PythonCode/AdvanceLSTM/BetterTrajVal/'

# Internal varaibles
# Set the different Occupancy Grid map and scene dimensions
occupancyMapWidth = 100
occupancyMapHeight = 600
OccupancyImageWidth = 512
OccupancyImageHeight = 2048
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
predTemporal = 100

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
    drop = 0.8
    epochs_drop = 5.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    if lrate < 0.000001:
        lrate = 0.000001
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
    alphaValue = 0.9   #0.5
    regVal = 0.0001

    # trajOutput, forward_state_h1, forward_state_c1, backward_state_h1, backward_state_c1 = Bidirectional(LSTM(256, return_sequences=True, return_state=True))(trajInput)
    trajOutput, state_c1, state_h1 = LSTM(256, return_sequences=True, return_state=True)(trajInput)
    trajOutput = TimeDistributed(LeakyReLU(alpha=alphaValue))(trajOutput)
    trajOutput = TimeDistributed(BatchNormalization())(trajOutput)
    trajOutput, state_c2, state_h2  = LSTM(128, return_sequences=True, return_state=True)(trajOutput)
    # trajOutput, forward_state_h2, forward_state_c2, backward_state_h2, backward_state_c2 = Bidirectional(LSTM(128, return_sequences=True, return_state=True))(trajInput)
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
    trajOutput = TimeDistributed(Dense(16))(trajOutput)
    trajOutput = TimeDistributed(LeakyReLU(alpha=alphaValue))(trajOutput)
    # trajOutput = Flatten()(trajOutput)

    # stateConcatenated1 = concatenate([state_c1, state_h1])
    # stateConcatenated2 = concatenate([state_c2, state_h2])
    stateConcatenated1 = [state_c1, state_h1]
    stateConcatenated2 = [state_c2, state_h2]

    decoder_lstm_1 = LSTM(256, return_sequences=True)

    decoder_lstm_2 = LSTM(128)

    decoderOutput = decoder_lstm_1(trajOutput,initial_state=stateConcatenated1)
    decoderOutput = TimeDistributed(LeakyReLU(alpha=alphaValue))(decoderOutput)
    decoderOutput = TimeDistributed(BatchNormalization())(decoderOutput)

    decoderOutput = decoder_lstm_2(decoderOutput,initial_state=stateConcatenated2)
    decoderOutput = LeakyReLU(alpha=alphaValue)(decoderOutput)
    decoderOutput = BatchNormalization()(decoderOutput)

    # trajOutputFinal = concatenate([trajOutput, totalState])
    # trajOutputFinal = concatenate([trajOutput, stateConcatenated2])
    # trajOutputFinal = concatenate([trajOutput, bothState])


    #trajOutputFinal = BatchNormalization()(trajOutputFinal)

    trajOutputFinal = Dense(1024)(decoderOutput)
    trajOutputFinal = LeakyReLU(alpha=alphaValue)(trajOutputFinal)
    trajOutputFinal = Dense(512)(decoderOutput)
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

    #model.save('encode.h5')

    print('Started preidiction.........')

    testFolderList = os.listdir(trainFolder)
    modifiedLonError = np.zeros((predTemporal))

    imageCount = 0

    #read input data for all the folders in the test folder

    for eachFolder in testFolderList:
        currentFrame = np.zeros((2048,512,3), dtype=np.uint8)
        currentFrame.fill(255)
        filePath = trainFolder + eachFolder + '/output.txt'
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
        inputTestArray = np.array(inputTestTraj).reshape(1, inputTemporal,2)

        # Read ground truth traj
        outputTrajStr = outputLines[0:predTemporal]
        outputTrajList = []
        for eachOutStr in outputTrajStr:
            outputX = float(eachOutStr.split(',')[0])
            outputY = float(eachOutStr.split(',')[1])
            outputTrajList.append([outputX,outputY])
        outputTrajArray = np.array(outputTrajList)

       

        # collect predictions
        outputModified = list()
        newInput = np.zeros((1,predTemporal,2))
        for t in range(predTemporal):
            # predict next pose
            yhat1 = model.predict(inputTestArray)
            # store prediction
            lastX = inputTestArray[:,-1,0]*OccupancyImageWidth
            lastY = inputTestArray[:,-1,1]*OccupancyImageHeight
            predX = yhat1[0][0]
            predY = yhat1[0][1]
            # if predY > 1024:
            #     print('In worong Update')
            #     predY = 1023

            if predY > (OccupancyImageHeight/2):
                print('In worong Update')
                predY = (OccupancyImageHeight/2) - 1

            # if (t==0):
            #     predX = yhat1[0][0][0][0]
            #     predY = yhat1[0][0][0][1]
            # else:
            #     predX = (yhat1[0][0][0][0]*0.5) + 0.5*lastPredXFilter
            #     predY = (yhat1[0][0][0][1]*0.5) + 0.5*lastPredYFilter
            
            #lastPredXFilter = predX
            #lastPredYFilter = predY

            #shiftX = lastX - predX
            #shiftY = lastY - predY
            #shiftY = lastY - outputTrajArray[t][1]

            firstX = inputTestArray[0,0,0]*OccupancyImageWidth
            firstY = inputTestArray[0,0,1]*OccupancyImageHeight

            secondX = inputTestArray[0,1,0]*OccupancyImageWidth
            secondY = inputTestArray[0,1,1]*OccupancyImageHeight

            newShiftX = firstX - secondX
            newShiftY = firstY - secondY

            inputTestArray[:,:-1,:] = inputTestArray[:,1:,:]
            inputTestArray[:,-1,:] = np.array([predX/OccupancyImageWidth, predY/OccupancyImageHeight])
            for pdx in range(0,int(inputTemporal)):
                inputTestArray[0,pdx,0] = ((inputTestArray[0,pdx,0]*OccupancyImageWidth) + newShiftX)/OccupancyImageWidth
                inputTestArray[0,pdx,1] = ((inputTestArray[0,pdx,1]*OccupancyImageHeight) + newShiftY)/OccupancyImageHeight

            if not outputModified:
                outputModified.append(np.array([yhat1[0][0], yhat1[0][1]]))
            else:
                lastPredX = outputModified[-1][0]
                lastPredY = outputModified[-1][1] 
                currentPredX = yhat1[0][0]
                currentPredY = yhat1[0][1]
                if currentPredY > (OccupancyImageHeight/2):
                    currentPredY = (OccupancyImageHeight/2) - 1
                predShiftX = (OccupancyImageWidth/2) - currentPredX
                predShiftY = (OccupancyImageHeight/2) - currentPredY
                newPredX = yhat1[0][0] #lastPredX - predShiftX
                newPredY = lastPredY - predShiftY
                outputModified.append(np.array([newPredX,newPredY]))
    
        resultModified = np.array(outputModified)

        # pts = np.array(outputTrajList, np.int32)
        # pts = pts.reshape((-1,1,2))
        # cv2.polylines(currentFrame,[pts],False,(0,255,0),2)

        # pts = np.array(outputModified, np.int32)
        # pts = pts.reshape((-1,1,2))
        # cv2.polylines(currentFrame,[pts],False,(0,0,255),2)

        # outputFileName = trajImageFolder + str(imageCount) + '.png'
        # cv2.imwrite(outputFileName, currentFrame)
        # imageCount = imageCount + 1



        print(str(eachFolder))
        # print(outputTrajArray)
        # print(resultModified)
        print('-------------------------')
        currentError = FrameLonError(outputTrajArray,resultModified)
        varError.append(currentError)
        modifiedLonError = modifiedLonError + currentError
        print(modifiedLonError/globalLonErrorCount)


    #plt.plot(originalLonError, label='Original')

    # with open('trajtotraj.txt', 'w') as f:
    #     for item in varError:
    #         f.write("%s\n" % item)

    plt.plot(modifiedLonError/globalLonErrorCount, label='Modified')
    plt.legend()
    plt.show()



 

   