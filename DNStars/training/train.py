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
    os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32"

os.environ["KERAS_BACKEND"] = "tensorflow"

if (platform.node() != 'viga'):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from keras.layers import Input, Dense, Convolution1D, MaxPooling1D, Flatten, merge, GaussianNoise, ZeroPadding1D
from keras.callbacks import ModelCheckpoint, Callback
from keras.models import Model, model_from_json
from keras.utils.visualize_util import plot as kerasPlot
import keras.optimizers
from keras.utils import np_utils
#from ipdb import set_trace as stop

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

    def __init__(self, root, noise, option):
        self.root = root
        self.nFeatures = 50
        self.kernelSize = 3
        self.poolLength = 2
        self.nLambda = np.asarray([2920, 2400, 1893])
        self.nLambdaNew = np.asarray([2944, 2400, 1920])
        self.batchSize = 512        
        self.option = option

        dims = np.asarray([7, 9, 5, 9, 5, 11, 11])

        self.nClasses = dims * 5

        self.noise = noise
        
        self.lower = np.asarray([])
        self.upper = np.asarray([])

        self.dataFile = "/scratch1/aasensio/deepLearning/DNStars/database/database.h5"

        f = h5py.File(self.dataFile, 'r')
        pars = f.get("parameters")        
        self.nModels, _ = pars.shape
        
        self.lower = np.min(pars, axis=0)
        self.upper = np.max(pars, axis=0)

        f.close()
        
        self.nTraining = int(self.nModels * 0.9)
        self.nValidation = int(self.nModels * 0.1)

        self.nBatchsPerEpochTraining = int(self.nTraining / self.batchSize)
        self.nBatchsPerEpochValidation = int(self.nValidation / self.batchSize)

        self.nTraining = self.nBatchsPerEpochTraining * self.batchSize
        self.nValidation = self.nBatchsPerEpochValidation * self.batchSize

        print("Training set: {0}".format(self.nTraining))
        print("   - Batch size: {0}".format(self.batchSize))
        print("   - Batches per epoch: {0}".format(self.nBatchsPerEpochTraining))

        print("Validation set: {0}".format(self.nValidation))
        print("   - Batch size: {0}".format(self.batchSize))
        print("   - Batches per epoch: {0}".format(self.nBatchsPerEpochValidation))

    def transformToCategorical(self, data, index):
        valuesInt = np.floor((data - self.lower[index]) / (self.upper[index] - self.lower[index]) * (self.nClasses[index]-1)).astype('int32')
        return np_utils.to_categorical(valuesInt, self.nClasses[index])

    def training_generator(self):
        f = h5py.File(self.dataFile, 'r')
        pars = f.get("parameters")
        flux = f.get("flux")

        while 1:        
            for i in range(self.nBatchsPerEpochTraining):

                outTrain = []

                for j in range(7):
                    outTrain.append(self.transformToCategorical(pars[i*self.batchSize:(i+1)*self.batchSize,j], j))
                
                continuum1 = nd.filters.uniform_filter1d(flux[i*self.batchSize:(i+1)*self.batchSize,0:2920], axis=1, size=30, mode='nearest')
                continuum2 = nd.filters.uniform_filter1d(flux[i*self.batchSize:(i+1)*self.batchSize,2920:2920+2400], axis=1, size=30, mode='nearest')
                continuum3 = nd.filters.uniform_filter1d(flux[i*self.batchSize:(i+1)*self.batchSize,2920+2400:], axis=1, size=30, mode='nearest')

                piece1 = np.atleast_3d(flux[i*self.batchSize:(i+1)*self.batchSize,0:2920] / continuum1).astype('float32')
                piece2 = np.atleast_3d(flux[i*self.batchSize:(i+1)*self.batchSize,2920:2920+2400] / continuum2).astype('float32')
                piece3 = np.atleast_3d(flux[i*self.batchSize:(i+1)*self.batchSize,2920+2400:] / continuum3).astype('float32')

                yield [piece1, piece2, piece3], outTrain

        f.close()

    def validation_generator(self):
        f = h5py.File(self.dataFile, 'r')
        pars = f.get("parameters")
        flux = f.get("flux")
        
        while 1:        
            for i in range(self.nBatchsPerEpochValidation):            

                outTrain = []

                for j in range(7):
                    outTrain.append(self.transformToCategorical(pars[self.nTraining+i*self.batchSize:self.nTraining+(i+1)*self.batchSize,j], j))
                            
                continuum1 = nd.filters.uniform_filter1d(flux[self.nTraining+i*self.batchSize:self.nTraining+(i+1)*self.batchSize,0:2920], axis=1, size=30, mode='nearest')
                continuum2 = nd.filters.uniform_filter1d(flux[self.nTraining+i*self.batchSize:self.nTraining+(i+1)*self.batchSize,2920:2920+2400], axis=1, size=30, mode='nearest')
                continuum3 = nd.filters.uniform_filter1d(flux[self.nTraining+i*self.batchSize:self.nTraining+(i+1)*self.batchSize,2920+2400:], axis=1, size=30, mode='nearest')

                piece1 = np.atleast_3d(flux[self.nTraining+i*self.batchSize:self.nTraining+(i+1)*self.batchSize,0:2920] / continuum1).astype('float32')
                piece2 = np.atleast_3d(flux[self.nTraining+i*self.batchSize:self.nTraining+(i+1)*self.batchSize,2920:2920+2400] / continuum2).astype('float32')
                piece3 = np.atleast_3d(flux[self.nTraining+i*self.batchSize:self.nTraining+(i+1)*self.batchSize,2920+2400:] / continuum3).astype('float32')

                yield [piece1, piece2, piece3], outTrain

        f.close()

    def validation_generator_prediction(self):
        f = h5py.File(self.dataFile, 'r')
        flux = f.get("flux")
        
        while 1:        
            for i in range(self.nBatchsPerEpochValidation):            

                batch = flux[self.nTraining+i*self.batchSize:self.nTraining+(i+1)*self.batchSize,:]                
                batch += np.random.normal(loc=0.0, scale=self.noise, size=batch.shape)
                continuum = np.copy(batch)

                for k in range(30):
                    continuum = nd.filters.uniform_filter1d(continuum, axis=1, size=30, mode='nearest')

                yield np.atleast_3d(batch / continuum).astype('float32')

        f.close()
        
    def defineNetwork(self):
        print("Setting up network...")

# Piece 1
        flux1 = Input(shape=(2920,1), name='flux1_input')
        x = GaussianNoise(sigma=self.noise)(flux1)
        x = ZeroPadding1D(padding=24)(x)
        

        x = Convolution1D(nb_filter=self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='conv1_1')(x)
        x = MaxPooling1D(pool_length=self.poolLength, name='pool1_1')(x)

        x = Convolution1D(nb_filter=self.nFeatures, filter_length=2*self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='conv2_1')(x)
        x = MaxPooling1D(pool_length=self.poolLength, name='pool2_1')(x)

        x = Convolution1D(nb_filter=self.nFeatures, filter_length=3*self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='conv3_1')(x)
        x = MaxPooling1D(pool_length=self.poolLength, name='pool3_1')(x)

        x = Convolution1D(nb_filter=self.nFeatures, filter_length=4*self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='conv4_1')(x)
        x = MaxPooling1D(pool_length=self.poolLength, name='pool4_1')(x)

        x = Convolution1D(nb_filter=self.nFeatures, filter_length=5*self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='conv5_1')(x)
        x = MaxPooling1D(pool_length=self.poolLength, name='pool5_1')(x)
        flux1_flat = Flatten(name='flat_1')(x)

# Piece 2
        flux2 = Input(shape=(2400,1), name='flux2_input')
        x = GaussianNoise(sigma=self.noise)(flux2)
        
        x = Convolution1D(nb_filter=self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='conv1_2')(x)
        x = MaxPooling1D(pool_length=self.poolLength, name='pool1_2')(x)

        x = Convolution1D(nb_filter=self.nFeatures, filter_length=2*self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='conv2_2')(x)
        x = MaxPooling1D(pool_length=self.poolLength, name='pool2_2')(x)

        x = Convolution1D(nb_filter=self.nFeatures, filter_length=3*self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='conv3_2')(x)
        x = MaxPooling1D(pool_length=self.poolLength, name='pool3_2')(x)

        x = Convolution1D(nb_filter=self.nFeatures, filter_length=4*self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='conv4_2')(x)
        x = MaxPooling1D(pool_length=self.poolLength, name='pool4_2')(x)

        x = Convolution1D(nb_filter=self.nFeatures, filter_length=5*self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='conv5_2')(x)
        x = MaxPooling1D(pool_length=self.poolLength, name='pool5_2')(x)
        flux2_flat = Flatten(name='flat_2')(x)

# Piece 3
        flux3 = Input(shape=(1893,1), name='flux3_input')
        x = GaussianNoise(sigma=self.noise)(flux3)
        x = ZeroPadding1D(padding=27)(x)
        

        x = Convolution1D(nb_filter=self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='conv1_3')(x)
        x = MaxPooling1D(pool_length=self.poolLength, name='pool1_3')(x)

        x = Convolution1D(nb_filter=self.nFeatures, filter_length=2*self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='conv2_3')(x)
        x = MaxPooling1D(pool_length=self.poolLength, name='pool2_3')(x)

        x = Convolution1D(nb_filter=self.nFeatures, filter_length=3*self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='conv3_3')(x)
        x = MaxPooling1D(pool_length=self.poolLength, name='pool3_3')(x)

        x = Convolution1D(nb_filter=self.nFeatures, filter_length=4*self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='conv4_3')(x)
        x = MaxPooling1D(pool_length=self.poolLength, name='pool4_3')(x)

        x = Convolution1D(nb_filter=self.nFeatures, filter_length=5*self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='conv5_3')(x)
        x = MaxPooling1D(pool_length=self.poolLength, name='pool5_3')(x)
        flux3_flat = Flatten(name='flat_3')(x)

        x = merge([flux1_flat, flux2_flat, flux3_flat], mode='concat', name='merge')
        
        labels = ['metals', 'C', 'N', 'O', 'alpha', 'log10vdop', 'Teff', 'logg']

        out = [None] * 7
        for i in range(7):
            out[i] = Dense(self.nClasses[i], activation='softmax', name='out_{0}'.format(labels[i]))(x)
        
        self.model = Model(input=[flux1, flux2, flux3], output=out)
        
        json_string = self.model.to_json()
        f = open('{0}_model.json'.format(self.root), 'w')
        f.write(json_string)
        f.close()

        kerasPlot(self.model, to_file='{0}_model.png'.format(self.root), show_shapes=True)

        with open('{0}_summary.txt'.format(self.root), 'w') as f:
            with redirect_stdout(f):
                self.model.summary()

    def compileNetwork(self):
        self.model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])

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
        
        self.metrics = self.model.fit_generator(self.training_generator(), self.nTraining, nb_epoch=nIterations, 
            callbacks=[self.checkpointer, self.history], validation_data=self.validation_generator(), nb_val_samples=self.nValidation,
            max_q_size=30)
        
        self.history.finalize()

    def predictCNN(self):
        print("Predicting validation data...")        
        out = self.model.predict_generator(self.validation_generator_prediction(), self.nValidation, max_q_size=30)

        print("Saving validation data...")
        with open("{0}_{1}_prob.pkl".format(self.root, self.noise), "wb") as outfile:
            pickle.dump(out, outfile, pickle.HIGHEST_PROTOCOL)

    def predictCNN2(self):
        print("Predicting validation data...")

        f = h5py.File(self.dataFile, 'r')
        flux = f.get("flux")

        batch = flux[0:1024,:]
        batch += np.random.normal(loc=0.0, scale=self.noise, size=batch.shape)
        continuum = np.copy(batch)
        
        for k in range(30):
            continuum = nd.filters.uniform_filter1d(continuum, axis=1, size=30, mode='nearest')


        inTest = np.atleast_3d(batch / continuum).astype('float32')
        
        out = self.model.predict(inTest, verbose=1)

        print("Saving validation data...")
        with open("{0}_{1}_prob.pkl".format(self.root, self.noise), "wb") as outfile:
            pickle.dump(out, outfile, pickle.HIGHEST_PROTOCOL)

if (__name__ == '__main__'):

    parser = argparse.ArgumentParser(description='Train/predict for spectra')
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
        out.compileNetwork()

    if (option == 'continue' or option == 'predict'):
        out.readNetwork()

    if (option == 'start' or option == 'continue'):
        out.compileNetwork()
        out.trainCNN(nEpochs)
    
    if (option == 'predict'):
        out.predictCNN2()