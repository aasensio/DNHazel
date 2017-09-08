import numpy as np
import matplotlib.pyplot as pl
import json
import os
import argparse
import time

from scipy.io import netcdf
from ipdb import set_trace as stop

import keras.backend as K
from keras.callbacks import CSVLogger, LearningRateScheduler, ModelCheckpoint
from keras.layers import Input, Lambda, Dense, Flatten, BatchNormalization, Activation, Conv1D, add, concatenate
from keras.layers.advanced_activations import PReLU
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.utils import plot_model
from sklearn.cluster import KMeans, AgglomerativeClustering
import pandas as pd
from contextlib import redirect_stdout

import pyhazel

def i0Allen(wavelength, muAngle):
    """
    Return the solar intensity at a specific wavelength and heliocentric angle
    wavelength: wavelength in angstrom
    muAngle: cosine of the heliocentric angle
    """
    C = 2.99792458e10
    H = 6.62606876e-27

    lambdaIC = 1e4 * np.asarray([0.20,0.22,0.245,0.265,0.28,0.30,0.32,0.35,0.37,0.38,0.40,0.45,0.50,0.55,0.60,0.80,1.0,1.5,2.0,3.0,5.0,10.0])
    uData = np.asarray([0.12,-1.3,-0.1,-0.1,0.38,0.74,0.88,0.98,1.03,0.92,0.91,0.99,0.97,0.93,0.88,0.73,0.64,0.57,0.48,0.35,0.22,0.15])
    vData = np.asarray([0.33,1.6,0.85,0.90,0.57, 0.20, 0.03,-0.1,-0.16,-0.05,-0.05,-0.17,-0.22,-0.23,-0.23,-0.22,-0.20,-0.21,-0.18,-0.12,-0.07,-0.07])

    lambdaI0 = 1e4 * np.asarray([0.20,0.22,0.24,0.26,0.28,0.30,0.32,0.34,0.36,0.37,0.38,0.39,0.40,0.41,0.42,0.43,0.44,0.45,0.46,0.48,0.50,0.55,0.60,0.65,0.70,0.75,\
        0.80,0.90,1.00,1.10,1.20,1.40,1.60,1.80,2.00,2.50,3.00,4.00,5.00,6.00,8.00,10.0,12.0])
    I0 = np.asarray([0.06,0.21,0.29,0.60,1.30,2.45,3.25,3.77,4.13,4.23,4.63,4.95,5.15,5.26,5.28,5.24,5.19,5.10,5.00,4.79,4.55,4.02,3.52,3.06,2.69,2.28,2.03,\
        1.57,1.26,1.01,0.81,0.53,0.36,0.238,0.160,0.078,0.041,0.0142,0.0062,0.0032,0.00095,0.00035,0.00018])
    I0 *= 1e14 * (lambdaI0 * 1e-8)**2 / C

    u = np.interp(wavelength, lambdaIC, uData)
    v = np.interp(wavelength, lambdaIC, vData)
    i0 = np.interp(wavelength, lambdaI0, I0)
    
    return (1.0 - u - v + u * muAngle + v * muAngle**2)* i0

def compute(pars):
    nPar, nSizeBlock = pars.shape

    stokesOut = np.zeros((4,128,nSizeBlock))

    nLambdaInput = 128
    GRIS_dispersion = 0.0362  # A/pix
    lowerLambda = 10828
    upperLambda = lowerLambda + GRIS_dispersion * nLambdaInput

    for i in range(nSizeBlock):
        tau, v, vth, a, B, theta, phi, mu = pars[:,i]

        synModeInput = 5
        nSlabsInput = 1

        B1Input = np.asarray([B, theta, phi])    
        B2Input = np.asarray([0.0,0.0,0.0])
        
        hInput = 3.e0

        tau1Input = tau
        tau2Input = 0.e0

        transInput = 1
        atomicPolInput = 1
        magoptInput = 1

        anglesInput = np.asarray([np.arccos(mu)*180/np.pi,0.0,0.0])

        lambdaAxisInput = np.linspace(lowerLambda-10829.0911, upperLambda-10829.0911, nLambdaInput)        

        dopplerWidthInput = vth
        dopplerWidth2Input = 0.e0

        dampingInput = a

        dopplerVelocityInput = v
        dopplerVelocity2Input = 0.e0

        ffInput = 0.e0
        betaInput = 1.0
        beta2Input = 1.0
        nbarInput = np.asarray([0.0,0.0,0.0,0.0])
        omegaInput = np.asarray([0.0,0.0,0.0,0.0])

        I0 = i0Allen(10830.0, mu)

        boundaryInput  = np.zeros((nLambdaInput,4))
        boundaryInput[:,0] = I0
        
        normalization = 0
        
        # Compute the Stokes parameters using many default parameters, using Allen's data
        [l, stokes, etaOutput, epsOutput] = pyhazel.synth(synModeInput, nSlabsInput, B1Input, B2Input, hInput, 
                                tau1Input, tau2Input, boundaryInput, transInput, atomicPolInput, magoptInput, anglesInput, 
                                nLambdaInput, lambdaAxisInput, dopplerWidthInput, dopplerWidth2Input, dampingInput, 
                                dopplerVelocityInput, dopplerVelocity2Input, ffInput, betaInput, beta2Input, nbarInput, omegaInput, normalization)

        stokesOut[:,:,i] = stokes

    return l, stokesOut

def residual(inputs, n_filters, activation, strides):
    x0 = Conv1D(n_filters, 1, padding='same', kernel_initializer='he_normal', strides=strides)(inputs)

    x = Conv1D(n_filters, 3, padding='same', kernel_initializer='he_normal', strides=strides)(inputs)
    x = BatchNormalization()(x)
    if (activation == 'prelu'):
        x = PReLU()(x)
    else:
        x = Activation(activation)(x)
    x = Conv1D(n_filters, 3, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = add([x0, x])

    return x

class kernel_mixture_network(object):

    def __init__(self, parsed):

        self.root = parsed['model']
        self.var = parsed['var']

        tmp = np.load("{0}_{1}_centers.npz".format(self.root, self.var))
        self.center_locs = tmp['center_locs']
        self.sigmas = tmp['sigmas']

        self.n_modes = len(self.sigmas)

        f = open("{0}_{1}_args.json".format(self.root, self.var), 'r')
        tmp = f.read()
        f.close()
        parsed = json.loads(tmp)

        self.infer_sigma = parsed['infer_sigma']

        self.oneDivSqrtTwoPI = 1.0 / np.sqrt(2.0*np.pi) # normalisation factor for gaussian.

        pyhazel.init()

        if (self.var == 'tau'):
            self.coeff = 2.0
            self.lower = 0.05
            self.upper = 3.0
            self.index = 0
        if (self.var == 'v'):
            self.coeff = 5.0
            self.lower = -5.0
            self.upper = 5.0
            self.index = 1
        if (self.var == 'vth'):
            self.coeff = 10.0
            self.lower = 5.0
            self.upper = 18.0
            self.index = 2
        if (self.var == 'a'):
            self.coeff = 0.5
            self.lower = 0.0
            self.upper = 0.5
            self.index = 3
        if (self.var == 'B'):
            self.coeff = 1000.0
            self.lower = 0.0
            self.upper = 1000.0
            self.index = 4
        if (self.var == 'thB'):
            self.coeff = 180.0
            self.lower = -180.0
            self.upper = 180.0
            self.index = 5
        if (self.var == 'phiB'):
            self.coeff = 180.0
            self.lower = -180.0
            self.upper = 180.0
            self.index = 6

    def gaussian_distribution(self, y, mu, sigma):
        result = (y - mu) / sigma
        result = - 0.5 * (result * result)
        return (K.exp(result) / sigma) * self.oneDivSqrtTwoPI

    def gaussian_distribution_np(self, y, mu, sigma):
        result = (y - mu) / sigma
        result = - 0.5 * (result * result)
        return (np.exp(result) / sigma) * self.oneDivSqrtTwoPI
    
    def mdn_loss_function_infer(self, args):
        y, weights, sigma_global = args
        result = self.gaussian_distribution(y, self.center_locs, self.sigmas * sigma_global) * weights
        result = K.sum(result, axis=1)
        result = - K.log(result)
        return K.mean(result)

    def mdn_loss_function(self, args):
        y, weights = args
        result = self.gaussian_distribution(y, self.center_locs, self.sigmas) * weights
        result = K.sum(result, axis=1)
        result = - K.log(result)
        return K.mean(result)

    def read_data(self):
        print("Generating validation data...")
        n_parameters = 8
        self.n_profiles = 100
        self.n_lambda = 128

        lower = np.asarray([0.05, -5.0, 5.0, 0.0, 0.0, 0.0, -180.0, 0.0])
        upper = np.asarray([3.0, 5.0, 18.0, 0.5, 1000.0, 180.0, 180.0, 1.0])


        rnd = np.random.rand(n_parameters, self.n_profiles)
        self.pars = (upper - lower)[:,None] * rnd + lower[:,None]

        l, stokes = compute(self.pars)

        stokes += np.random.normal(loc=0.0, scale=1e-4, size=stokes.shape)

        scaling = np.array([1.0,0.01,0.01,0.1])

        self.train = []
        for i in range(4):
            self.train.append((stokes[i,:,0:self.n_profiles] / scaling[i]).T.reshape((self.n_profiles, self.n_lambda, 1)).astype('float32'))
        self.train.append(np.zeros(self.n_profiles).astype('float32'))
        self.train.append(self.pars[-1,0:self.n_profiles].reshape((self.n_profiles, 1)).astype('float32'))

    def build_estimator(self):

# Inputs
        input_I = Input(shape=(self.n_lambda,1), name='stokes_I')        

        input_Q = Input(shape=(self.n_lambda,1), name='stokes_Q')

        input_U = Input(shape=(self.n_lambda,1), name='stokes_U')        

        input_V = Input(shape=(self.n_lambda,1), name='stokes_V')

        input_x = concatenate([input_I,input_Q,input_U,input_V])
        

        y_true = Input(shape=(1,), name='y_true')
        mu_input = Input(shape=(1,), name='mu_input')

# Neural network
        x = Conv1D(64, 3, padding='same', kernel_initializer='he_normal', name='conv_1')(input_x)
        x = PReLU()(x)

        kernels = [64, 64, 64]

        for i in range(3):
            x = residual(x, kernels[i], 'prelu', strides=2)
    
        intermediate = Flatten(name='flat')(x)
        intermediate_conv = concatenate([intermediate, mu_input], name='FC')

        weights = Dense(self.n_modes, activation='softmax', name='weights')(intermediate_conv)

        if (self.infer_sigma == 'yes'):
            sigma_global = Dense(1, activation='softplus', name='sigma')(intermediate_conv)

# Definition of the loss function
            loss = Lambda(self.mdn_loss_function_infer, output_shape=(1,), name='loss')([y_true, weights, sigma_global])

        else:
# Definition of the loss function
            loss = Lambda(self.mdn_loss_function, output_shape=(1,), name='loss')([y_true, weights])
        
        self.model = Model(inputs=[input_I,input_Q,input_U,input_V, y_true, mu_input], outputs=[loss])
    
# Compile with the loss weight set to None, so it will be omitted
        self.model.load_weights("{0}_{1}_best.h5".format(self.root, self.var))

# Now generate a second network that ends up in the weights for later evaluation
        if (self.infer_sigma == 'yes'):
            self.model_weights = Model(inputs=self.model.input,
                                 outputs=[self.model.get_layer('weights').output, self.model.get_layer('sigma').output])
        else:
            self.model_weights = Model(inputs=self.model.input,
                                 outputs=self.model.get_layer('weights').output)
                       
    def forward_network(self):
        print("Reading network...")
        self.build_estimator()

        y = np.linspace(self.lower,self.upper,300).reshape((300,1))
        
        if (self.infer_sigma == 'yes'):
            weights, sigma = self.model_weights.predict(self.train)
        else:
            weights = self.model_weights.predict(self.train)
            sigma = np.ones((self.n_profiles,1))

        prob = np.zeros((self.n_profiles,300))

        for i in range(self.n_profiles):
            result = self.gaussian_distribution_np(y / self.coeff, self.center_locs, self.sigmas * sigma[i,0]) * weights[i,:]
            prob[i,:] = np.sum(result, axis=-1)

        f, ax = pl.subplots(nrows=3, ncols=3)
        ax = ax.flatten()
        for i in range(9):
            ax[i].plot(y, prob[i,:])
            ax[i].axvline(self.pars[self.index,i])

        pl.show()

        f, ax = pl.subplots()
        ax.plot(y-self.pars[self.index,:], prob.T, color='C0', alpha=0.1)

        pl.show()

        n = 100
        out = np.zeros((self.n_profiles,n))
        for i in range(self.n_profiles):
            ind = np.random.choice(self.n_modes, p=weights[i,:], size=n)
            out[i,:] = self.coeff * np.random.normal(loc=self.center_locs[ind], scale=self.sigmas[ind] * sigma[i,0])

        inp = np.repeat(self.pars[self.index,:],n)

        f, ax = pl.subplots()
        ax.plot(out.flatten(), inp, '.', color='C0', alpha=0.1)
        pl.show()

        stop()

    def predict_density(self, x_test):
        y = np.linspace(-10,10,300).reshape((300,1))
        weights = self.model.predict(x_test)
        result = self.gaussian_distribution(torch.unsqueeze(y,1), self.center_locs, self.sigmas) * weights
        result = torch.sum(result, dim=1)
        return y.data.numpy(), result
    
if (__name__ == '__main__'):


    parser = argparse.ArgumentParser(description='Predict for KMN')
    parser.add_argument('-m','--model', help='Output files', required=True)
    parser.add_argument('-b','--batch_size', help='Batch size', default=256)
    parser.add_argument('-i','--infer_sigma', help='Infer sigma', default='no')
    parser.add_argument('-v','--var', help='Variable to train', choices=['tau','v','vth','a','B','thB','phiB','thB_LOS','phiB_LOS'], 
        default='tau', required=True)

    parsed = vars(parser.parse_args())

    out = kernel_mixture_network(parsed)
    out.read_data()
    
    out.forward_network()