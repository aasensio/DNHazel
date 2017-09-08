import numpy as np
import matplotlib.pyplot as pl
import platform
import os
from astropy.io import fits
from ipdb import set_trace as stop
from astropy.io import fits
import scipy.io
import time
import argparse
import h5py
from matplotlib.widgets import Slider
import matplotlib.animation as manimation
from tqdm import tqdm
import scipy.misc as mi
import congrid

if (platform.node() == 'viga'):
    os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32"

os.environ["KERAS_BACKEND"] = "tensorflow"

if (platform.node() != 'viga'):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
import keras.backend.tensorflow_backend as ktf

import sys
sys.path.append('../training')

import models as nn_model

def contrast(x):
    return 100 * np.std(x) / np.mean(x)

class deep_network(object):

    def __init__(self, networks, depths, activation, n_filters, validation_sets, normalization, output_files):

# Only allocate needed memory
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        session = tf.Session(config=config)
        ktf.set_session(session)

        self.networks = networks
        self.activation = activation
        self.n_filters = n_filters
        self.depth = depths
        self.input_file_images = validation_sets
        self.normalization = normalization
        self.output_files = output_files


    def define_network(self, root, depth, nx=50, ny=50):
        print("Setting up network...")

        self.model = nn_model.keepsize(self.nx, self.ny, 0.0, depth, activation=self.activation, n_filters=self.n_filters)
        
        print("Loading weights...")
        self.model.load_weights("../training/networks/{0}_{1}_weights.hdf5".format(root, depth))
    
    def gen_data_validation(self, nx=50, ny=50):

        self.nx = nx
        self.ny = ny

        print("Predicting validation data...")

        for i in range(len(self.networks)):

            self.median_HMI, self.median_SST = self.normalization[i]

            f = h5py.File(self.input_file_images[i], 'r')

            input_validation = np.zeros((100,self.nx,self.ny,1), dtype='float32')
        
            input_validation[:,:,:,:] = f['imHMI'][0:100,:,:,0:1].astype('float32') / self.median_HMI
            output_validation = f['imSST'][0:100,:,:,0:1].astype('float32') / self.median_SST

            self.define_network(self.networks[i], self.depth[i])
            start = time.time()
            out = self.model.predict(input_validation)
            end = time.time()
            print("Prediction took {0} seconds...".format(end-start))
            np.savez(self.output_files[i], input_validation, output_validation, out)
            
                    
if (__name__ == '__main__'):

    pl.close('all')

    networks = ['keepsize_x2_noise', 'keepsize_x2_blos_2']
    depths = [5, 5]
    filters_exit = [64, 64]
    validation_sets = ['/net/viga/scratch1/cdiazbas/DATABASE/database_validation_x2_PSF2.h5', '/net/viga/scratch1/cdiazbas/DATABASE/database_validation_x2_BLOS.h5']
    normalization = [np.loadtxt('/net/vena/scratch/Dropbox/GIT/DeepLearning/hmi_super/training/normalization.txt'), np.array([1.0,1.0])]
    output_files = ['continuum.npz', 'blos.npz']
    out = deep_network(networks, depths, activation='relu', n_filters=64, validation_sets=validation_sets, normalization=normalization, output_files=output_files)
    # out.gen_movie_blos_validation()
    out.gen_data_validation()


    # networks = ['keepsize_x2_PSF1', 'keepsize_x2_PSF1', 'keepsize_x2_PSF2_64out', 'keepsize_x2_PSF2']
    # depths = [15,10,5,5,5]
    # filters_exit = [256, 256, 64, 256]
    # out = deep_network(networks, depths, filters_exit, activation='relu', n_filters=64)
    # out.gen_movie()

