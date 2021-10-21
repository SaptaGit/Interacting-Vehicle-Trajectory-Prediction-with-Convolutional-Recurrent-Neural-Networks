from PIL import Image
import glob
import numpy as np
from keras.models import load_model
from keras import backend as K



occupancyMapWidth = 100
occupancyMapHeight = 600
OccupancyImageWidth = 128
OccupancyImageHeight = 1024
imageOriginX = OccupancyImageWidth/2
imageOriginY = OccupancyImageHeight/2
lonResolution = OccupancyImageHeight/occupancyMapHeight 
channel = 1
sample = 1
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



modelFilePath = '/home/saptarshi/PythonCode/AdvanceLSTMServer/TrainedModels/OGMConvLSTMConcat.h5'

model = load_model(modelFilePath, custom_objects={'euclidean_distance_loss': euclidean_distance_loss, 'euclidean_distance_loss' : euclidean_distance_loss})

primaryImageFiles = np.zeros((sample,temporal,OccupancyImageHeight,OccupancyImageWidth,channel))
image_list = []

for ldx, filename in enumerate(glob.glob('/home/saptarshi/PythonCode/AdvanceLSTM/maptest/28/*.png')): #assuming gif
    ldx = int(filename.split('/')[-1].split('.')[0]) - 841
    im = np.asarray(Image.open(filename))
    primaryImageFiles[0,ldx,:,:,0] = im/255

res = model.predict(primaryImageFiles)

print(res)
