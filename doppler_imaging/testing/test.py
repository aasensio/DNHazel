import numpy as np
import matplotlib.pyplot as pl
from astropy.io import fits
import platform
import os
import json
import argparse
import h5py
from contextlib import redirect_stdout
import copy
from ipdb import set_trace as stop

if (platform.node() == 'viga'):
    os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32"

os.environ["KERAS_BACKEND"] = "tensorflow"

if (platform.node() != 'viga'):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from keras.callbacks import ModelCheckpoint, Callback, LearningRateScheduler
from keras.models import model_from_json
from keras.optimizers import Adam
from keras.utils import plot_model
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf
import time

import sys
sys.path.append('../training')

import models as nn_model

class deep_network(object):

    def __init__(self, root):

# Only allocate needed memory
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        session = tf.Session(config=config)
        ktf.set_session(session)


        self.root = root        
        self.noise = 0.0
        self.batch_size = 16

        self.normalizations = np.load('../training/normalization.npy')

        self.input_validation = "/net/viga/scratch1/deepLearning/doppler_imaging/database/validation_stars.h5"
        
        f = h5py.File(self.input_validation, 'r')
        self.n_validation = len(f['modulus'])
        f.close()    

        self.batchs_per_epoch_validation = int(self.n_validation / self.batch_size)

    def validation_generator(self):

        f = h5py.File(self.input_validation, 'r')
        
        while 1:        
            for i in range(self.batchs_per_epoch_validation):

                stokes_test = np.vstack(f['stokesv'][i*self.batch_size:(i+1)*self.batch_size]).astype('float32')
                _, n_lambda = stokes_test.shape
                n_viewpoints = int(n_lambda / 150.0)
                stokes_test = stokes_test.reshape((self.batch_size, n_viewpoints, 150, 1))

                max = np.max(np.abs(stokes_test), axis=(1,2))
                stokes_test /= max[:,None,None,:]


                # modulus_test = np.hstack(f['modulus'][i*self.batch_size:(i+1)*self.batch_size] / 3e3).astype('float32')

                yield stokes_test

        f.close()

    def define_network(self):
        print("Setting up network...")

        self.model = nn_model.zdi(150, 0.0, activation='relu', n_filters=64, l2_reg=1e-7)
        self.model.load_weights("{0}_weights.hdf5".format(self.root))
        
    def test(self, batch_size):
        print("Testing network...")

        i = 0

        f = h5py.File(self.input_validation, 'r')
        stokes_test = np.vstack(f['stokesv'][i*batch_size:(i+1)*batch_size] / 1e-4).astype('float32')
        _, n_lambda = stokes_test.shape
        n_viewpoints = int(n_lambda / 150.0)
        stokes_test = stokes_test.reshape((batch_size, n_viewpoints, 150, 1))

        tmp = np.vstack(f['alpha'][i*self.batch_size:(i+1)*self.batch_size])
        alpha_test = (tmp - self.normalizations[0,:][None,:]) / (self.normalizations[1,:][None,:] - self.normalizations[0,:][None,:]) - 0.5

        # modulus_test = np.hstack(f['modulus'][i*batch_size:(i+1)*batch_size] / 3e3).astype('float32')
            
        return self.model.predict(stokes_test, batch_size=1), alpha_test[:,0:5]

if (__name__ == '__main__'):
    
    root = '../training/networks/test'
    out = deep_network(root)

    out.define_network()
    out_nn, out_db = out.test(16)
    pl.plot(out_db.T, out_nn.T,'.')
    pl.plot([-0.4,0.4], [-0.4,0.4])
    pl.show()
