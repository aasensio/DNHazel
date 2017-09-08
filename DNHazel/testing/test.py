import numpy as np
import platform
import json
import sys
import os
import copy
import argparse
import time

os.environ["KERAS_BACKEND"] = "tensorflow"


from keras.callbacks import ModelCheckpoint, Callback, LearningRateScheduler
from keras.models import model_from_json
from keras.utils import plot_model
from keras.optimizers import Adam
from scipy.io import netcdf
import keras.backend.tensorflow_backend as ktf
import tensorflow as tf

sys.path.append('../training')

import models as nn_model

from ipdb import set_trace as stop

def log_sum_exp(x, axis=None):
    """Log-sum-exp trick implementation"""
    x_max = ktf.max(x, axis=axis, keepdims=True)
    return ktf.log(ktf.sum(ktf.exp(x - x_max), axis=axis, keepdims=True))+x_max

class deep_network(object):

    def __init__(self):

# Only allocate needed memory
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        session = tf.Session(config=config)
        ktf.set_session(session)

        self.root = '../training/cnns/test'
        self.batch_size = 32
        self.fraction_training = 0.9
        self.noise = 0.0
        self.activation = 'relu'
        self.depth = 5
        self.n_kernels = 64
        self.n_mixture = 8
        self.c = 9  # Number of variables
        self.l2_reg = 1e-7

        self.lower = np.asarray([0.05, -5.0, 5.0, 0.0, 0.0, 0.0, -180.0, 0.0, -180.0])
        self.upper = np.asarray([3.0, 5.0, 18.0, 0.5, 1000.0, 180.0, 180.0, 180.0, 180.0]) 

    def read_data(self):
        print("Reading data...")
        self.f = netcdf.netcdf_file('/net/viga/scratch1/deepLearning/DNHazel/database/database_mus_1000000.db', 'r')
        self.stokes = self.f.variables['stokes'][:]
        self.parameters = self.f.variables['parameters'][:]
        self.n_lambda = len(self.stokes[0,:,0])
        self.n_training = int(self.fraction_training * len(self.stokes[0,0,:]))

        mu = self.parameters[7,:]
        thB = self.parameters[5,:] * np.pi / 180.0
        phiB = self.parameters[6,:] * np.pi / 180.0

        cosThB = mu * np.cos(thB) + np.sqrt(1.0-mu**2) * np.sin(thB) * np.cos(phiB)
        sinThB = np.sqrt(1.0 - cosThB**2)

        cosPhiB = (mu * np.sin(thB) * np.cos(phiB) - np.sqrt(1.0-mu**2) * np.cos(thB)) / sinThB
        sinPhiB = np.sin(thB) * np.sin(phiB) / sinThB

        ThB = np.arctan2(sinThB, cosThB) * 180.0 / np.pi
        PhiB = np.arctan2(sinPhiB, cosPhiB) * 180.0 / np.pi

        self.inTrain = []
        self.inTrain.append(self.stokes[:,:,0:10].T.reshape((10, self.n_lambda, 4)).astype('float32'))
        self.inTrain.append(self.parameters[-1,0:10].reshape((10, 1)).astype('float32'))
        
        self.outTrain = []
        for i in range(7):
            self.outTrain.append((self.parameters[i,0:self.n_training] - self.lower[i]) / (self.upper[i] - self.lower[i]).astype('float32'))

# Add outputs for LOS angles
        outTrain = (ThB[0:self.n_training] - self.lower[7]) / (self.upper[7] - self.lower[7]).astype('float32')
        self.outTrain.append(outTrain)

        outTrain = (PhiB[0:self.n_training] - 0.001 - self.lower[8]) / (self.upper[8] - self.lower[8]).astype('float32')
        self.outTrain.append(outTrain)

        self.outTrain = np.array(self.outTrain).T

        self.f.close()

    def define_network(self):
        
        self.model = nn_model.network(self.n_lambda, self.depth, noise=self.noise, activation=self.activation, n_filters=self.n_kernels, l2_reg=self.l2_reg)
        

    def predict(self):
        parameters = self.model.predict(self.inTrain)

        components = np.reshape(parameters,[-1, 2*9 + 1, self.n_mixture])
        
        mu = components[:, 0:9, :]
        sigma = components[:, 9:18, :]
        alpha = components[:, 18, :]
        
        return mu, sigma, alpha


if (__name__ == '__main__'):

    
    out = deep_network()
    out.read_data()
    out.define_network()
    mu, sigma, alpha = out.predict()