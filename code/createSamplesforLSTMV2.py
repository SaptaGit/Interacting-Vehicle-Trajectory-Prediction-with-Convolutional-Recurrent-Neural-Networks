import os
from PIL import Image
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed, Conv2D, MaxPooling2D, Flatten
import matplotlib.pyplot as plt
import sys
import cv2
import gc
#from keras import backend as K

#K.set_image_dim_ordering('th')

imageWidth = 256
imageHeight = 1024
channel = 1
temporal = 30

def euclidean_distance_loss(y_true, y_pred):
    """
    Euclidean distance loss
    https://en.wikipedia.org/wiki/Euclidean_distance
    :param y_true: TensorFlow/Theano tensor
    :param y_pred: TensorFlow/Theano tensor of the same shape as y_true
    :return: float
    """
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

def generate_data(directory, batch_size):
    """Replaces Keras' native ImageDataGenerator."""
    i = 0
    folderList = sorted(os.listdir(directory), key=int)
    imageType = '.png'
    outputFile = 'output.txt'

    while True:
        sampleInput = []
        sampleOutput = []
        for b in range(batch_size):
        #gather input
            fileList = os.listdir(directory + folderList[i])
            imagesList = [i for i in fileList if i.endswith(imageType)]
            sortedImageFiles = sorted(imagesList, key=lambda a: int(a.split(".")[0]) )
            imageInput = []
            for ldx,imageFile in enumerate(sortedImageFiles):
                imageData = cv2.imread(directory + folderList[i] + '/' +  imageFile, 0)/255
                imageInput.append(imageData)
            sampleInput.append(imageInput)

        #gather output
            # extract the future position from the output.txt file as output for each sample
            f = open(directory + folderList[i] + '/' +  outputFile,'r')
            outputLines = f.read().splitlines()
            f.close()
            outList = []
            for mdx,outputValue in enumerate(outputLines):
                outputX = float(outputValue.split(',')[0])
                outputY = float(outputValue.split(',')[1])
                outList.append([outputX,outputY])
            sampleOutput.append(outList)     
            i = i+1
        X = np.array(sampleInput).reshape(batch_size,temporal,imageHeight,imageWidth, channel)
        y = np.array(sampleOutput)
        if(i >= len(folderList) - 1):
            i=0
        print('after Yeild........')
        print('i=' + str(i))
        #gc.collect()
        yield X,y 

# def LoadSamples():
#     #Get all the folder of samples
#     folderPath = '/home/saptarshi/PythonCode/AdvanceLSTM/OccupancyMapsV2/'
#     folderList = []
#     for i,j,y in sorted(os.walk(folderPath)):
#         folderList.append(i)

#     # Pop out the first item as this is holding the parent folder
#     folderList.pop(0)

#     #allSample = []
#     imageType = '.png'
#     outputFile = 'output.txt'

#     count = 0
#     sampleCount = len(folderList)
#     inputArray = np.empty((10,30,1024,256))
#     outputArray = np.empty((sampleCount,30,2))

#     for kdx,folder in enumerate(folderList):
#         count = count + 1
#         print('sample :' + str(count))
#         # Extract all the images for one sample as input
#         fileList = os.listdir(folder)
#         imagesList = [i for i in fileList if i.endswith(imageType)]
#         sortedImageFiles = sorted(imagesList, key=lambda a: int(a.split(".")[0]) )
#         #eachSample = []
#         #sampleInput = []
#         #sampleOutput = []
#         for ldx,imageFile in enumerate(sortedImageFiles):
#             #img = Image.open(folder + '/' +  imageFile)
#             #img.load()
#             #imageData = np.asarray(img,dtype=np.uint8)
#             #img = cv2.imread(folder + '/' +  imageFile)
#             #sampleInput.append(cv2.imread(folder + '/' +  imageFile))
#             inputArray[kdx,ldx,:,:] = cv2.imread(folder + '/' +  imageFile)/255
#             #img.close()
#         # extract the future position from the output.txt file as output for each sample
#         f = open(folder + '/' +  outputFile,'r')
#         outputLines = f.read().splitlines()
#         f.close()
#         for mdx,outputValue in enumerate(outputLines):
#             outputX = float(outputValue.split(',')[0])
#             outputY = float(outputValue.split(',')[1])
#             outputArray[kdx,mdx,:] = np.array([outputX,outputY])
#             #sampleOutput.append([outputX,outputY])
        
#        # eachSample.append(sampleInput)
#         #eachSample.append(sampleOutput)
#        # allSample.append(eachSample)
#         print('sampleInput : ' + str(sys.getsizeof(sampleInput)))
#         print('sampleOutput : ' + str(sys.getsizeof(sampleOutput)))
#         print('eachSample : ' + str(sys.getsizeof(eachSample)))
#         print('allSample : ' + str(sys.getsizeof(allSample)))
#         #eachSample = []
#         #sampleInput = []
#         #sampleOutput = []

#     #return allSample
#     return inputArray, outputArray

# def SplitTraininputOutput(samples):
#     trainInput = []
#     trainOutput = []
#     for sample in samples:
#         inputVal = sample[0]
#         outputVal = np.array(sample[1])
#         trainInput.append(inputVal)
#         trainOutput.append(outputVal)
#     trainInputArray = np.array(trainInput)
#     trainOutputArray = np.array(trainOutput)
#     return trainInputArray,trainOutputArray


if __name__ == '__main__':

    # trainSamples = LoadSamples()
    # numberOfSamples = len(trainSamples)
    # trainX,trainY = SplitTraininputOutput(trainSamples)
    # trainX = trainX.reshape(numberOfSamples,30,1024,256,1)

    #trainX, trainY = LoadSamples()
    dataFolder = '/home/saptarshi/PythonCode/AdvanceLSTM/OccupancyMapsV2/'
    numberOfSamples = len(os.listdir(dataFolder))

    # Basic LSTM structure with Conv2D
    model = Sequential()
    # define CNN model
    model.add(TimeDistributed(Conv2D(32, (3,3), activation='relu', padding='same'), batch_input_shape=(numberOfSamples,30,1024,256,1)))
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
    gen = generate_data(dataFolder, batch_size)
    model.fit_generator(gen, steps_per_epoch=len(os.listdir(dataFolder)) // batch_size, epochs=20)

    model.save('AdvanceLSTM.h5')
    training_loss = history.history['loss']
    epoch_count = range(1, len(training_loss) + 1)
    plt.plot(epoch_count, training_loss, 'r--')
    plt.legend(['Training Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()







