import os
from PIL import Image
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed, Conv2D, MaxPooling2D, Flatten
import matplotlib.pyplot as plt
#from keras import backend as K

#K.set_image_dim_ordering('th')

def euclidean_distance_loss(y_true, y_pred):
    """
    Euclidean distance loss
    https://en.wikipedia.org/wiki/Euclidean_distance
    :param y_true: TensorFlow/Theano tensor
    :param y_pred: TensorFlow/Theano tensor of the same shape as y_true
    :return: float
    """
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

def LoadSamples():
    #Get all the folder of samples
    folderPath = '/home/saptarshi/PythonCode/AdvanceLSTM/OccupancyMapsV2/'
    folderList = []
    for i,j,y in sorted(os.walk(folderPath)):
        folderList.append(i)

    # Pop out the first item as this is holding the parent folder
    folderList.pop(0)

    allSample = []
    imageType = '.jpeg'
    outputFile = 'output.txt'
    eachSample = []
    sampleInput = []
    sampleOutput = []
    count = 0
    for folder in folderList:
        count = count + 1
        print('sample :' + str(count))
        # Extract all the images for one sample as input
        fileList = os.listdir(folder)
        imagesList = [i for i in fileList if i.endswith(imageType)]
        sortedImageFiles = sorted(imagesList, key=lambda a: int(a.split(".")[0]) )
        for imageFile in sortedImageFiles:
            img = Image.open(folder + '/' +  imageFile)
            img.load()
            imageData = np.asarray(img,dtype=np.uint8)
            sampleInput.append(imageData)
        # extract the future position from the output.txt file as output for each sample
        outputLines = open(folder + '/' +  outputFile,'r').read().splitlines()
        for outputValue in outputLines:
            outputX = float(outputValue.split(',')[0])
            outputY = float(outputValue.split(',')[1])
            sampleOutput.append([outputX,outputY])
        
        eachSample.extend(sampleInput)
        eachSample.extend(sampleOutput)
        allSample.extend(eachSample)
        sampleInput.clear()
        sampleOutput.clear()
        eachSample.clear()


    return allSample

def SplitTraininputOutput(samples):
    trainInput = []
    trainOutput = []
    for sample in samples:
        inputVal = sample[0]
        outputVal = np.array(sample[1])
        trainInput.append(inputVal)
        trainOutput.append(outputVal)
    trainInputArray = np.array(trainInput)
    trainOutputArray = np.array(trainOutput)
    return trainInputArray,trainOutputArray


if __name__ == '__main__':

    trainSamples = LoadSamples()
    numberOfSamples = len(trainSamples)
    trainX,trainY = SplitTraininputOutput(trainSamples)
    trainX = trainX.reshape(numberOfSamples,30,1024,256,1)

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
    history = model.fit(trainX, trainY, epochs=8, verbose=1)
    model.save('AdvanceLSTM.h5')
    training_loss = history.history['loss']
    epoch_count = range(1, len(training_loss) + 1)
    plt.plot(epoch_count, training_loss, 'r--')
    plt.legend(['Training Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()







