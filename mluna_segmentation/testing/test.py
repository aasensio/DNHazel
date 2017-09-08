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
import scipy.io as io
from skimage import exposure

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

        f = open('{0}_args.json'.format(root), 'r')
        tmp = json.load(f)
        f.close()

        self.noise = float(tmp['noise'])
        self.depth = int(tmp['depth'])
        self.n_filters = int(tmp['kernels'])
        self.activation = tmp['activation']    
        self.batch_size = int(tmp['batchsize'])
        self.l2_reg = float(tmp['l2_regularization'])
        self.root = root        
        self.noise = 0.0
        self.batch_size = 16
        self.nx = 500
        self.ny = 500

        
    def define_network(self):
        print("Setting up network...")

        # self.model = nn_model.keepsize(self.nx, self.ny, self.noise, self.depth, activation=self.activation, n_filters=self.n_filters, l2_reg=self.l2_reg)
        self.model = nn_model.encdec(self.nx, self.ny, self.noise, self.depth, activation=self.activation, n_filters=self.n_filters, l2_reg=self.l2_reg)
        self.model.load_weights("{0}_weights.hdf5".format(self.root))
        
    def test(self):
        print("Testing network...")        
        im = np.zeros((1,self.nx,self.ny,1))
        self.im_ha = np.load('../database/images_ha.npy')
        mask_ha = np.load('../database/mask_ha.npy')
        self.mn, self.mx = np.load('../database/normalization.npy')

        im[0,:,:,0] = (self.im_ha[1][500:1000,400:900] - self.mn) / (self.mx - self.mn)
            
        return self.model.predict(im.astype('float32'), batch_size=1), mask_ha[1][500:1000,400:900]

if (__name__ == '__main__'):
    
    root = '../training/networks/test_5'

    out = deep_network(root)
    out.define_network()
    nn, mask = out.test()    
    nn_mask = np.squeeze(np.argmax(nn, axis=-1))

    f, ax = pl.subplots(nrows=1, ncols=3, figsize=(14,6))

    im = ax[0].imshow((out.im_ha[1][500:1000,400:900] - out.mn) / (out.mx - out.mn))
    pl.colorbar(im, ax=ax[0])

    im = ax[1].imshow(mask)
    pl.colorbar(im, ax=ax[1])

    im = ax[2].imshow(nn_mask)
    pl.colorbar(im, ax=ax[2])

    pl.show()