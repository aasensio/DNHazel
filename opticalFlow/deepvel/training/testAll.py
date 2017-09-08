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

from keras.layers import Input, Dense, Convolution2D, Flatten, merge, MaxPooling2D, UpSampling2D, Cropping2D
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
        self.nx = 156
        self.ny = 156
        self.n_times = 3
        
        
    def defineNetwork(self):
        print("Setting up network...")

        inputs = Input(shape=(self.nx, self.ny, self.n_times))
        conv1a_down = Convolution2D(32, 3, 3, activation='relu', border_mode='valid')(inputs)
        conv1b_down = Convolution2D(64, 3, 3, activation='relu', border_mode='valid')(conv1a_down)
        crop1 = Cropping2D(cropping=((40, 40), (40, 40)))(conv1b_down)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1b_down)

        conv2a_down = Convolution2D(64, 3, 3, activation='relu', border_mode='valid')(pool1)
        conv2b_down = Convolution2D(128, 3, 3, activation='relu', border_mode='valid')(conv2a_down)
        crop2 = Cropping2D(cropping=((16, 16), (16, 16)))(conv2b_down)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2b_down)

        conv3a_down = Convolution2D(128, 3, 3, activation='relu', border_mode='valid')(pool2)
        conv3b_down = Convolution2D(256, 3, 3, activation='relu', border_mode='valid')(conv3a_down)
        crop3 = Cropping2D(cropping=((4, 4), (4, 4)))(conv3b_down)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3b_down)

        conv4a_down = Convolution2D(256, 3, 3, activation='relu', border_mode='valid')(pool3)
        conv4b_down = Convolution2D(512, 3, 3, activation='relu', border_mode='valid')(conv4a_down)

        up3 = UpSampling2D(size=(2, 2))(conv4b_down)
        merge3 = merge([up3, crop3], mode='concat', concat_axis=3)

        conv3a_up = Convolution2D(256, 3, 3, activation='relu', border_mode='valid')(merge3)
        conv3b_up = Convolution2D(256, 3, 3, activation='relu', border_mode='valid')(conv3a_up)

        up2 = UpSampling2D(size=(2, 2))(conv3b_up)
        merge2 = merge([up2, crop2], mode='concat', concat_axis=3)

        conv2a_up = Convolution2D(128, 3, 3, activation='relu', border_mode='valid')(merge2)
        conv2b_up = Convolution2D(128, 3, 3, activation='relu', border_mode='valid')(conv2a_up)

        up1 = UpSampling2D(size=(2, 2))(conv2b_up)
        merge1 = merge([up1, crop1], mode='concat', concat_axis=3)

        conv1a_up = Convolution2D(128, 3, 3, activation='relu', border_mode='valid')(merge1)
        conv1b_up = Convolution2D(128, 3, 3, activation='relu', border_mode='valid')(conv1a_up)

        final = Convolution2D(6, 1, 1, activation='linear', border_mode='valid')(conv1b_up)

        self.model = Model(input=inputs, output=final)

        self.model.load_weights("{0}_weights.hdf5".format(self.root))     

    def validation_generator(self):
        f_images = h5py.File('/net/duna/scratch1/aasensio/deepLearning/opticalFlow/database/database_images_validation.h5', 'r')
        images = f_images.get("intensity")    
        
        while 1:        
            for i in range(30):

                input_validation = images[10*i:10*i+10,:,:,:].astype('float32')     

                yield input_validation

        f_images.close()

    def predict_validation(self):
        print("Predicting validation data...")

        tmp = np.load('/net/viga/scratch/Dropbox/GIT/DeepLearning/opticalFlow/database/normalizations.npz')

        meanI, stdI, minx, miny, maxx, maxy = tmp['arr_0'], tmp['arr_1'], tmp['arr_2'], tmp['arr_3'], tmp['arr_4'], tmp['arr_5']

        f_images = h5py.File('/net/duna/scratch1/aasensio/deepLearning/opticalFlow/database/database_images_validation.h5', 'r')
        f_velocity = h5py.File('/net/duna/scratch1/aasensio/deepLearning/opticalFlow/database/database_velocity_validation.h5', 'r')

        im = f_images.get("intensity")[:]
        v = f_velocity.get("velocity")[:]

        # im -= np.mean(meanI[1:])
        # im /= np.mean(stdI[1:])

        im = im.astype('float32')

        minx = np.min(minx[:,1:], axis=1)
        maxx = np.max(maxx[:,1:], axis=1)

        miny = np.min(miny[:,1:], axis=1)
        maxy = np.max(maxy[:,1:], axis=1)

        out = self.model.predict_generator(self.validation_generator(), 300)
        
        out[:,:,:,0] *= (maxx[0] - minx[0])
        out[:,:,:,0] += minx[0]
        out[:,:,:,1] *= (maxy[0] - miny[0])
        out[:,:,:,1] += miny[0]

        out[:,:,:,2] *= (maxx[1] - minx[1])
        out[:,:,:,2] += minx[1]
        out[:,:,:,3] *= (maxy[1] - miny[1])
        out[:,:,:,3] += miny[1]

        out[:,:,:,4] *= (maxx[2] - minx[2])
        out[:,:,:,4] += minx[2]
        out[:,:,:,5] *= (maxy[2] - miny[2])
        out[:,:,:,5] += miny[2]

        v[:,:,:,0] *= (maxx[0] - minx[0])
        v[:,:,:,0] += minx[0]
        v[:,:,:,1] *= (maxy[0] - miny[0])
        v[:,:,:,1] += miny[0]

        v[:,:,:,2] *= (maxx[1] - minx[1])
        v[:,:,:,2] += minx[1]
        v[:,:,:,3] *= (maxy[1] - miny[1])
        v[:,:,:,3] += miny[1]

        v[:,:,:,4] *= (maxx[2] - minx[2])
        v[:,:,:,4] += minx[2]
        v[:,:,:,5] *= (maxy[2] - miny[2])
        v[:,:,:,5] += miny[2]

        pl.close('all')
        
        minv = -0.8
        maxv = 0.8

        np.random.seed(123)
        index = np.random.permutation(300)

        label = [1, 0.1, 0.01]

        for loop in range(3):
            f, ax = pl.subplots(nrows=3, ncols=5, figsize=(18,10))
            for ind in range(3):
                res = ax[ind,0].imshow(im[index[ind],44:-44,44:-44,0])
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
    
    out = trainDNNFull('cnns/unetAll1')
    out.defineNetwork()
    out.predict_validation()