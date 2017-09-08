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

from keras.layers import Input, Dense, Convolution2D, Flatten, merge, MaxPooling2D, UpSampling2D, Cropping2D, Deconvolution2D, Activation, Lambda
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
        self.nx = 50
        self.ny = 50
        self.n_times = 2
        self.n_filters = 64
        self.batch_size = 32        
        self.n_conv_layers = 10
        self.stride = 1
        self.skip_frequency = 2
        self.n_diversity = 2
        self.input_file_images_validation = "/net/duna/scratch1/aasensio/deepLearning/phaseDiv/database/database_images_validation.h5"
        
        
    def readNetwork(self):
        print("Reading previous network...")
                
        f = open('{0}_model.json'.format(self.root), 'r')
        json_string = f.read()
        f.close()

        json_string = json_string.replace('"output_shape": [null', '"output_shape": [%d' % 32)

        self.model = model_from_json(json_string)
        self.model.load_weights("{0}_weights.hdf5".format(self.root))

    def defineNetwork(self):
        print("Setting up network...")

        conv = [None] * self.n_conv_layers
        deconv = [None] * self.n_conv_layers

        inputs = Input(shape=(self.nx, self.ny, self.n_diversity))
        conv[0] = Convolution2D(self.n_filters, 3, 3, activation='relu', subsample=(self.stride,self.stride), border_mode='same', init='he_normal')(inputs)
        for i in range(self.n_conv_layers-1):
            conv[i+1] = Convolution2D(self.n_filters, 3, 3, activation='relu', subsample=(self.stride,self.stride), border_mode='same', init='he_normal')(conv[i])

        deconv[0] = Deconvolution2D(self.n_filters, 3, 3, activation='relu', output_shape=(self.batch_size, self.nx, self.ny,self.n_filters), subsample=(self.stride,self.stride), border_mode='same', init='he_normal')(conv[-1])
        for i in range(self.n_conv_layers-1):
            if (i % self.skip_frequency == 0):
                x = Deconvolution2D(self.n_filters, 3, 3, output_shape=(self.batch_size,self.nx, self.ny,self.n_filters), activation='relu', subsample=(self.stride,self.stride), border_mode='same', init='he_normal')(deconv[i])                
                x = merge([conv[self.n_conv_layers-i-2], x], mode='sum')
                deconv[i+1] = Activation('relu')(x)

            else:
                deconv[i+1] = Deconvolution2D(self.n_filters, 3, 3, output_shape=(self.batch_size,self.nx, self.ny,self.n_filters), activation='relu', subsample=(self.stride,self.stride), border_mode='same', init='he_normal')(deconv[i])

        x = Deconvolution2D(1, 1, 1, output_shape=(self.batch_size,self.nx, self.ny, 1), activation='linear', subsample=(self.stride,self.stride), border_mode='same', init='he_normal')(deconv[-1])

        focused = Lambda(lambda x: x[:,:,:,0:1], output_shape=(self.nx, self.ny, 1))(inputs)

        final = merge([x, focused], 'sum')
        self.model = Model(input=inputs, output=final)

        self.model.load_weights("{0}_weights.hdf5".format(self.root))

    def validation_generator(self):
        f_images = h5py.File(self.input_file_images_validation, 'r')
        images = f_images.get("intensity")        
        
        while 1:        
            for i in range(1):

                input_validation = images[i*self.batch_size:(i+1)*self.batch_size,:,:,1:3].astype('float32')

                yield input_validation

        f_images.close()        

    def predict_validation(self):
        print("Predicting validation data...")        

        f_images = h5py.File('/net/duna/scratch1/aasensio/deepLearning/phaseDiv/database/database_images_validation.h5', 'r')

        im = f_images.get("intensity")[:]
        
        im = im.astype('float32')

        out = self.model.predict_generator(self.validation_generator(), 32)
        
        pl.close('all')   

        np.random.seed(123)
        index = np.random.permutation(30)

        f, ax = pl.subplots(nrows=3, ncols=5, figsize=(18,10))
        for ind in range(5):

            res = ax[0,ind].imshow(im[index[ind],:,:,0], cmap=pl.cm.gray)
            pl.colorbar(res, ax=ax[0,ind])            

            res = ax[1,ind].imshow(im[index[ind],:,:,1], cmap=pl.cm.gray)
            pl.colorbar(res, ax=ax[1,ind])            

            res = ax[2,ind].imshow(out[index[ind],:,:,0], cmap=pl.cm.gray)
            pl.colorbar(res, ax=ax[2,ind])            

        pl.tight_layout()

        pl.show()

        pl.savefig("{0}_prediction_tau.png".format(self.root))

        stop()
            
if (__name__ == '__main__'):
    
    out = trainDNNFull('cnns/test')
    out.defineNetwork()
    out.predict_validation()
