import numpy as np
import h5py
import numpy as np
import platform
import os
import json
import sys
import argparse
import scipy.ndimage as nd
import pickle
from contextlib import redirect_stdout
from ipdb import set_trace as stop

if (platform.node() == 'viga'):
    os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32,cuda.root=/usr/local/cuda"

if (platform.node() == 'Andress-MacBook-Pro.local'):
    os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32"


os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from keras.layers import Input, Dense, Convolution1D, MaxPooling1D, Flatten, merge, GaussianNoise, BatchNormalization, AtrousConvolution1D
from keras.callbacks import ModelCheckpoint, Callback
from keras.models import Model, model_from_json
from keras.utils.visualize_util import plot as kerasPlot
from keras.optimizers import Adam
from keras.utils import np_utils

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

    def __init__(self, root, noise, option):
        self.root = root
        self.nFeatures = 64
        self.kernelSize = 5
        self.poolLength = 2
        self.nLambda = 112
        self.batchSize = 64
        self.nClasses = [50] * 12
        self.noise = noise
        self.option = option

        self.labels = ['T0', 'T1', 'T2', 'vmic', 'B0', 'B1', 'v0', 'v1', 'thB0', 'thB1', 'chiB0', 'chiB1']

        self.n_pars = len(self.labels)

# BField, theta, chi, vmac, damping, B0, B1, doppler, kl
        self.lower = np.asarray([-3000.0, -1500.0, -3000.0, 0.0, 0.0, 0.0, -7.0, -7.0, 0.0, 0.0, 0.0, 0.0], dtype='float32')
        self.upper = np.asarray([3000.0, 3000.0, 5000.0, 4.0, 3000.0, 3000.0, 7.0, 7.0, 180.0, 180.0, 180.0, 180.0], dtype='float32')
        
        self.dataFile = "/net/viga/scratch1/deepLearning/DNSIR/database/database_sir.h5"

        f = h5py.File(self.dataFile, 'r')
        pars = f.get("parameters")
        stokes = f.get("stokes")
        self.nModels, _ = pars.shape
                
        self.nTraining = int(self.nModels * 0.9)
        self.nValidation = int(self.nModels * 0.1)

# Standardize Stokes parameters
        # std_values = np.std(np.abs(stokes[0:self.nTraining,:,:]),axis=0)
        # stokes /= std_values[None,:,:]        
        minim = np.min(stokes,axis=(0,2))[None,:,None]
        maxim = np.max(stokes,axis=(0,2))[None,:,None]
        stokes = (stokes - minim) / (maxim - minim)

        stokes = np.transpose(stokes, [0,2,1])

# Save normalization values
        np.savez('{0}_normalization.npz'.format(self.root), minim, maxim)
        
        print("Training set: {0}".format(self.nTraining))

        print("Validation set: {0}".format(self.nValidation))

        self.inTrain = np.atleast_3d(stokes[0:self.nTraining,:,:]).astype('float32')        
        self.inTest = np.atleast_3d(stokes[self.nTraining:,:,:]).astype('float32')

        self.outTrain = []
        for i in range(self.n_pars):
            outTrain = np.floor((pars[0:self.nTraining, i] - self.lower[i]) / (self.upper[i] - self.lower[i]) * self.nClasses[i]).astype('int32')            
            self.outTrain.append(np_utils.to_categorical(outTrain, self.nClasses[i]))

        self.outTest = []
        for i in range(self.n_pars):
            outTest = np.floor((pars[self.nTraining:, i] - self.lower[i]) / (self.upper[i] - self.lower[i]) * self.nClasses[i]).astype('int32')
            self.outTest.append(np_utils.to_categorical(outTest, self.nClasses[i]))

        f.close()
        
        
    def defineNetwork(self):
        print("Setting up network...")

# Stokes IQUV
        stokes_input = Input(shape=(self.nLambda,4), name='stokes_input')
        x = Convolution1D(nb_filter=self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='conv_1')(stokes_input)        
        x = Convolution1D(nb_filter=self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='conv_2a')(x)
        x = Convolution1D(nb_filter=self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='conv_2b')(x)
        x = Convolution1D(nb_filter=self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='conv_2c')(x)
        x = MaxPooling1D(pool_length=self.poolLength, name='pool_1')(x)
        x = Convolution1D(nb_filter=2*self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='conv_3a')(x)
        x = Convolution1D(nb_filter=2*self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='conv_3b')(x)
        x = Convolution1D(nb_filter=2*self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='conv_3c')(x)
        x = Convolution1D(nb_filter=2*self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='conv_3d')(x)
        x = MaxPooling1D(pool_length=self.poolLength, name='pool_2')(x)
        x = Convolution1D(nb_filter=4*self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='conv_4a')(x)
        x = Convolution1D(nb_filter=4*self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='conv_4b')(x)
        x = Convolution1D(nb_filter=4*self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='conv_4c')(x)
        x = Convolution1D(nb_filter=4*self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='conv_4d')(x)
        x = MaxPooling1D(pool_length=self.poolLength, name='pool_3')(x)
        x = Convolution1D(nb_filter=8*self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='conv_5a')(x)
        x = Convolution1D(nb_filter=8*self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='conv_5b')(x)
        x = Convolution1D(nb_filter=8*self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='conv_5c')(x)
        x = Convolution1D(nb_filter=8*self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='conv_5d')(x)
        x = MaxPooling1D(pool_length=self.poolLength, name='pool_4')(x)
        x = Flatten(name='flat')(x)

        x = Dense(200, activation='relu', name='FC2')(x)


        out = [None] * self.n_pars
        for i in range(self.n_pars):
            out[i] = Dense(self.nClasses[i], activation='softmax', name='out_{0}'.format(self.labels[i]))(x)

        self.model = Model(input=stokes_input, output=out)
        
        json_string = self.model.to_json()
        f = open('{0}_model.json'.format(self.root), 'w')
        f.write(json_string)
        f.close()

        with open('{0}_summary.txt'.format(self.root), 'w') as f:
            with redirect_stdout(f):
                self.model.summary()

        kerasPlot(self.model, to_file='{0}_model.png'.format(self.root), show_shapes=True)

    def compileNetwork(self):
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def readNetwork(self):
        print("Reading previous network...")
                
        f = open('{0}_model.json'.format(self.root), 'r')
        json_string = f.read()
        f.close()

        self.model = model_from_json(json_string)
        self.model.load_weights("{0}_weights.hdf5".format(self.root))

    def trainCNN(self, nIterations):
        print("Training network...")        
        
        self.checkpointer = ModelCheckpoint(filepath="{0}_weights.hdf5".format(self.root), verbose=1, save_best_only=True)

# Recover losses from previous run
        if (self.option == 'continue'):
            with open("{0}_loss.json".format(self.root), 'r') as f:
                losses = json.load(f)
        else:
            losses = []

        self.history = LossHistory(self.root, losses)        
        self.metrics = self.model.fit(self.inTrain, self.outTrain, nb_epoch=nIterations, batch_size=self.batchSize,
            validation_data=(self.inTest, self.outTest), callbacks=[self.checkpointer, self.history])
        
        self.history.finalize()

    def predictCNN(self):
        print("Predicting validation data...")

        f = h5py.File(self.dataFile, 'r')        
        stokes = f.get("stokes")

        inTest = []
        for i in range(4):
            prof = stokes[self.nTraining:,:,i]
            prof += np.random.normal(loc=0.0, scale=self.noise, size=prof.shape)
            inTest.append(np.atleast_3d(prof).astype('float32'))
        
        out = self.model.predict(inTest, verbose=1)

        print("Saving validation data...")
        with open("{0}_{1}_prob.pkl".format(self.root, self.noise), "wb") as outfile:
            pickle.dump(out, outfile, pickle.HIGHEST_PROTOCOL)

if (__name__ == '__main__'):

    parser = argparse.ArgumentParser(description='Train/predict for Milne-Eddington')
    parser.add_argument('-o','--out', help='Output files')
    parser.add_argument('-e','--epochs', help='Number of epochs', default=10)
    parser.add_argument('-n','--noise', help='Noise to add during training/prediction', default=0.0)
    parser.add_argument('-a','--action', help='Action', choices=['start', 'continue', 'predict'], required=True)
    parsed = vars(parser.parse_args())

    root = parsed['out']
    nEpochs = int(parsed['epochs'])
    option = parsed['action']
    noise = float(parsed['noise'])

    out = trainDNNFull(root, noise, option)

    if (option == 'start'):           
        out.defineNetwork()     

    if (option == 'continue' or option == 'predict'):
        out.readNetwork()

    if (option == 'start' or option == 'continue'):
        out.compileNetwork()
        out.trainCNN(nEpochs)
    
    if (option == 'predict'):
        out.predictCNN()