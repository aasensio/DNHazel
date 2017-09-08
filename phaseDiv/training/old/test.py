import numpy as np
import matplotlib.pyplot as pl
import h5py
import platform
import os
import json
import sys
import argparse
import scipy.ndimage as nd
import pickle
import scipy.io as io
from ipdb import set_trace as stop

if (platform.node() == 'viga'):
    os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32"

os.environ["KERAS_BACKEND"] = "tensorflow"

if (platform.node() != 'viga'):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from keras.layers import Input, Dense, Convolution2D, Flatten, merge, MaxPooling2D, UpSampling2D, Cropping2D, AtrousConvolution2D
from keras.callbacks import ModelCheckpoint, Callback
from keras.models import Model, model_from_json
from keras.utils.visualize_util import plot as kerasPlot
import keras.optimizers
from keras.utils import np_utils

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / N 

class LossHistory(Callback):
    def __init__(self, root, losses):
        self.root = root        
        self.losses = losses

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs)
        with open("{0}_loss.json".format(self.root), 'w') as f:
            json.dump(self.losses, f)

    def finalize(self):
        pass

class trainDNNFull(object):

    def __init__(self, root):

        self.root = root
        self.nx = 128
        self.ny = 128
        self.n_times = 3
        
        
    def defineNetwork(self):
        print("Setting up network...")

        input1 = Input(shape=(self.nx, self.ny, 2))
        input2 = Input(shape=(self.nx, self.ny, 1))

        conv = AtrousConvolution2D(64, 3, 3, atrous_rate=(1,1), activation='relu', border_mode='same', init='he_normal')(input1)
        for i in range(5):            
            conv = AtrousConvolution2D(64, 3, 3, atrous_rate=(2**(i+1),2**(i+1)), activation='relu', border_mode='same', init='he_normal')(conv)                

        final = Convolution2D(1, 3, 3, activation='linear', border_mode='same', init='he_normal')(conv)

        final = merge([final, input2], mode='sum')

        self.model = Model(input=[input1, input2], output=final)

        self.model.load_weights("{0}_weights.hdf5".format(self.root))     

    def validation_generator(self):
        f_images = h5py.File('/net/duna/scratch1/aasensio/deepLearning/phaseDiv/database/database_validation.h5', 'r')
        images = f_images.get("intensity")
        
        while 1:        
            for i in range(3):

                input_validation1 = images[i*10:(i+1)*10,:,:,1:3].astype('float32')
                input_validation2 = images[i*10:(i+1)*10,:,:,1:2].astype('float32')
                
                yield [input_validation1, input_validation2]

        f_images.close()

    def predict_validation(self):
        print("Predicting validation data...")
        

        f_images = h5py.File('/net/duna/scratch1/aasensio/deepLearning/phaseDiv/database/database_validation.h5', 'r')

        im = f_images.get("intensity")[:].astype('float32')
        
        out = self.model.predict_generator(self.validation_generator(), 30)

        f, ax = pl.subplots(nrows=3, ncols=5, figsize=(15,10))
        for i in range(5):
            ax[0,i].imshow(im[i,:,:,0])
            ax[1,i].imshow(im[i,:,:,1])
            ax[2,i].imshow(out[i,:,:,0])

        pl.show()
        
        
        stop()

if (__name__ == '__main__'):
    
    out = trainDNNFull('cnns/test')
    out.defineNetwork()
    out.predict_validation()