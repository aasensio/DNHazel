import numpy as np
import platform
import json
import sys
import os
import copy
import argparse
import time

from scipy.io import netcdf
from ipdb import set_trace as stop

import keras.backend as K
from keras.callbacks import CSVLogger, LearningRateScheduler, ModelCheckpoint
from keras.layers import Input, Lambda, Dense, Flatten, BatchNormalization, Activation, Conv1D, add, concatenate
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.utils import plot_model
from sklearn.cluster import KMeans, AgglomerativeClustering
import pandas as pd
from contextlib import redirect_stdout

def residual(inputs, n_filters, activation, strides):
    x0 = Conv1D(n_filters, 1, padding='same', kernel_initializer='he_normal', strides=strides)(inputs)

    x = Conv1D(n_filters, 3, padding='same', kernel_initializer='he_normal', strides=strides)(inputs)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)    
    x = Conv1D(n_filters, 3, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = add([x0, x])

    return x

class kernel_mixture_network(object):

    def __init__(self, parsed):

        self.root = parsed['model']
        self.var = parsed['var']

        self.lower = np.asarray([0.05, -5.0, 5.0, 0.0, 0.0, 0.0, -180.0, 0.0, -180.0])
        self.upper = np.asarray([3.0, 5.0, 18.0, 0.5, 1000.0, 180.0, 180.0, 180.0, 180.0])

        tmp = np.load("{0}_{1}_centers.npz".format(self.root, self.var))
        self.center_locs = tmp['center_locs']
        self.sigmas = tmp['sigmas']

        self.n_modes = len(self.sigmas)

        self.oneDivSqrtTwoPI = 1.0 / np.sqrt(2.0*np.pi) # normalisation factor for gaussian.

    def gaussian_distribution(self, y, mu, sigma):
        result = (y - mu) / sigma
        result = - 0.5 * (result * result)
        return (K.exp(result) / sigma) * self.oneDivSqrtTwoPI

    def gaussian_distribution_np(self, y, mu, sigma):
        result = (y - mu) / sigma
        result = - 0.5 * (result * result)
        return (np.exp(result) / sigma) * self.oneDivSqrtTwoPI
    
    def mdn_loss_function(self, args):
        y, weights = args
        result = self.gaussian_distribution(y, self.center_locs, self.sigmas) * weights
        result = K.sum(result, axis=1)
        result = - K.log(result)
        return K.mean(result)

    def read_data(self):
        print("Reading data...")
        self.f = netcdf.netcdf_file('/net/viga/scratch1/deepLearning/DNHazel/database/database_mus_1000000.db', 'r')
        self.stokes = self.f.variables['stokes'][:]
        self.parameters = self.f.variables['parameters'][:]
        self.n_lambda = len(self.stokes[0,:,0])
        self.n_training = 1000 #int(self.fraction_training * len(self.stokes[0,0,:]))

        mu = self.parameters[7,:]
        thB = self.parameters[5,:] * np.pi / 180.0
        phiB = self.parameters[6,:] * np.pi / 180.0

        cosThB = mu * np.cos(thB) + np.sqrt(1.0-mu**2) * np.sin(thB) * np.cos(phiB)
        sinThB = np.sqrt(1.0 - cosThB**2)

        cosPhiB = (mu * np.sin(thB) * np.cos(phiB) - np.sqrt(1.0-mu**2) * np.cos(thB)) / sinThB
        sinPhiB = np.sin(thB) * np.sin(phiB) / sinThB

        ThB = np.arctan2(sinThB, cosThB) * 180.0 / np.pi
        PhiB = np.arctan2(sinPhiB, cosPhiB) * 180.0 / np.pi

# Add training data, which include the Stokes parameters, the value of the output variable and mu
        self.train = []
        self.train.append(self.stokes[:,:,0:self.n_training].T.reshape((self.n_training, self.n_lambda, 4)).astype('float32'))
        if (self.var == 'tau'):
            var = self.parameters[0,0:self.n_training].reshape((self.n_training, 1)) / 2.0
        if (self.var == 'v'):
            var = self.parameters[1,0:self.n_training].reshape((self.n_training, 1)) / 5.0
        if (self.var == 'vth'):
            var = self.parameters[2,0:self.n_training].reshape((self.n_training, 1)) / 10.0
        if (self.var == 'a'):
            var = self.parameters[3,0:self.n_training].reshape((self.n_training, 1)) / 0.5
        if (self.var == 'B'):
            var = self.parameters[4,0:self.n_training].reshape((self.n_training, 1)) / 1000.0
        if (self.var == 'thB'):
            var = thB[0:self.n_training].reshape((self.n_training, 1)) / np.pi
        if (self.var == 'phiB'):
            var = phiB[0:self.n_training].reshape((self.n_training, 1)) / np.pi
        if (self.var == 'thB_LOS'):
            var = ThB[0:self.n_training].reshape((self.n_training, 1)) / np.pi
        if (self.var == 'phiN_LOS'):
            var = PhiB[0:self.n_training].reshape((self.n_training, 1)) / np.pi

        self.train.append(var.astype('float32'))
        self.train.append(self.parameters[-1,0:self.n_training].reshape((self.n_training, 1)).astype('float32'))

    def build_estimator(self):

# Inputs
        input_x = Input(shape=(self.n_lambda,4), name='stokes_input')
        y_true = Input(shape=(1,), name='y_true')
        mu_input = Input(shape=(1,), name='mu_input')

# Neural network
        x = Conv1D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv_1')(input_x)

        for i in range(3):
            x = residual(x, 64*(i+1), 'relu', strides=2)
    
        intermediate = Flatten(name='flat')(x)
        intermediate_conv = concatenate([intermediate, mu_input], name='FC')

# Output weights
        weights = Dense(self.n_modes, activation='softmax', name='weights')(intermediate_conv)

# Definition of the loss function
        loss = Lambda(self.mdn_loss_function, output_shape=(1,), name='loss')([y_true, weights])
        
        self.model = Model(inputs=[input_x, y_true, mu_input], outputs=[loss])
        #self.model.add_loss(loss)
    
# Compile with the loss weight set to None, so it will be omitted
        #self.model.compile(loss=[None], loss_weights=[None], optimizer=Adam(lr=self.lr))
        self.model.load_weights("{0}_{1}_best.h5".format(self.root, self.var))

# Now generate a second network that ends up in the weights for later evaluation
        self.model_weights = Model(inputs=self.model.input,
                                 outputs=self.model.get_layer('weights').output)
                       
    def forward_network(self):
        print("Reading network...")
        self.build_estimator()

        y = np.linspace(0.0,2.0,300).reshape((300,1))
        
        weights = self.model_weights.predict(self.train)

        prob = np.zeros((self.n_training,300))

        for i in range(self.n_training):
            result = self.gaussian_distribution_np(y, self.center_locs, self.sigmas) * weights[i,:]
            prob[i,:] = np.sum(result, axis=-1)

        stop()

    def predict_density(self, x_test):
        y = np.linspace(-10,10,300).reshape((300,1))
        weights = self.model.predict(x_test)
        result = self.gaussian_distribution(torch.unsqueeze(y,1), self.center_locs, self.sigmas) * weights
        result = torch.sum(result, dim=1)
        return y.data.numpy(), result
    
if (__name__ == '__main__'):


    parser = argparse.ArgumentParser(description='Predict for KMN')
    parser.add_argument('-o','--model', help='Output files', required=True)
    parser.add_argument('-b','--batch_size', help='Batch size', default=256)
    parser.add_argument('-v','--var', help='Variable to train', choices=['tau','v','vth','a','B','thB','phiB','thB_LOS','phiB_LOS'], 
        default='tau', required=True)

    parsed = vars(parser.parse_args())

    out = kernel_mixture_network(parsed)
    out.read_data()
    
    out.forward_network()