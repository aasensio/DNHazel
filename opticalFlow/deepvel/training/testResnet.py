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

from keras.layers import Input, Dense, Convolution2D, Flatten, merge, MaxPooling2D, UpSampling2D, Cropping2D, Deconvolution2D, Activation
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
        
        
    def readNetwork(self):
        print("Reading previous network...")
                
        f = open('{0}_model.json'.format(self.root), 'r')
        json_string = f.read()
        f.close()

        self.model = model_from_json(json_string)
        self.model.load_weights("{0}_weights.hdf5".format(self.root))

    def validation_generator(self):
        f_images = h5py.File('/net/duna/scratch1/aasensio/deepLearning/opticalFlow/database/database_images_validation.h5', 'r')
        images = f_images.get("intensity")    
        
        while 1:        
            for i in range(1):

                input_validation = images[i:i+self.batch_size,:,:,:].astype('float32')     

                yield input_validation

        f_images.close()

    def predict_validation(self):
        print("Predicting validation data...")        

        f_images = h5py.File('/net/duna/scratch1/aasensio/deepLearning/opticalFlow/database/database_images_validation.h5', 'r')
        f_velocity = h5py.File('/net/duna/scratch1/aasensio/deepLearning/opticalFlow/database/database_velocity_validation.h5', 'r')

        im = f_images.get("intensity")[:]
        v = f_velocity.get("velocity")[:]

        im = im.astype('float32')

        out = self.model.predict_generator(self.validation_generator(), 32)
        
        pl.close('all')
        
        minv = 0.0
        maxv = 1.0

        np.random.seed(123)
        index = np.random.permutation(30)

        label = [1, 0.1, 0.01]

        for loop in range(3):
            f, ax = pl.subplots(nrows=3, ncols=5, figsize=(18,10))
            for ind in range(3):

                minv = np.min([v[index[ind],:,:,2*loop],v[index[ind],:,:,2*loop+1]])
                maxv = np.max([v[index[ind],:,:,2*loop],v[index[ind],:,:,2*loop+1]])

                res = ax[ind,0].imshow(im[index[ind],:,:,0])
                pl.colorbar(res, ax=ax[ind,0])            

                res = ax[ind,1].imshow(v[index[ind],:,:,2*loop], vmin=minv, vmax=maxv)
                pl.colorbar(res, ax=ax[ind,1])

                res = ax[ind,2].imshow(v[index[ind],:,:,2*loop+1], vmin=minv, vmax=maxv)
                pl.colorbar(res, ax=ax[ind,2])

                res = ax[ind,3].imshow(out[index[ind],:,:,2*loop], vmin=minv, vmax=maxv)
                pl.colorbar(res, ax=ax[ind,3])

                res = ax[ind,4].imshow(out[index[ind],:,:,2*loop+1], vmin=minv, vmax=maxv)
                pl.colorbar(res, ax=ax[ind,4])

                ax[ind,1].set_title(r'vx ($\tau$={0})'.format(label[loop]))
                ax[ind,2].set_title(r'vy ($\tau$={0})'.format(label[loop]))
                ax[ind,3].set_title(r'vx(CNN) ($\tau$={0})'.format(label[loop]))
                ax[ind,4].set_title(r'vy(CNN) ($\tau$={0})'.format(label[loop]))


            ax[0,0].set_title('Time 1')
            ax[1,0].set_title('Time 2')
            ax[2,0].set_title('Time 3')
        
            pl.tight_layout()

            pl.show()

            pl.savefig("{0}_prediction_tau_{1}.png".format(self.root, label[loop]))

        stop()
            
if (__name__ == '__main__'):
    
    out = trainDNNFull('cnns/resnet2')
    out.readNetwork()
    out.predict_validation()
