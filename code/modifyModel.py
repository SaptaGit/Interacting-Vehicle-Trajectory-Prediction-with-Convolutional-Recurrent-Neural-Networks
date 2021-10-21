import os
import PIL
from PIL import Image
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed, Conv2D, MaxPooling2D, Flatten
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
import cv2
import sys
import skimage

def euclidean_distance_loss(y_true, y_pred):
    """
    Euclidean distance loss
    https://en.wikipedia.org/wiki/Euclidean_distance
    :param y_true: TensorFlow/Theano tensor
    :param y_pred: TensorFlow/Theano tensor of the same shape as y_true
    :return: float
    """
    result = K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))
    return result

model = load_model('/home/saptarshi/PythonCode/AdvanceLSTM/TrainedModels/MatlabTest.h5', custom_objects={'euclidean_distance_loss': euclidean_distance_loss})
model.summary()
model.save('matlab.h5')
