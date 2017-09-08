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

from keras.layers import Input, Dense, Convolution1D, MaxPooling1D, Flatten, merge, GaussianNoise
from keras.callbacks import ModelCheckpoint, Callback
from keras.models import Model, model_from_json
from keras.utils.visualize_util import plot as kerasPlot
import keras.optimizers
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
        self.nFeatures = 100
        self.kernelSize = 3
        self.poolLength = 2
        self.nLambda = 50
        self.batchSize = 256
        self.nClasses = [50, 50, 50, 50, 10, 20, 20, 20, 20]
        self.noise = noise
        self.option = option

                                # BField, theta, chi, vmac, damping, B0, B1, doppler, kl
        self.lower = np.asarray([0.0,      0.0,   0.0, -7.0, 0.0,  0.15, 0.15, 0.20,  1.0], dtype='float32')
        self.upper = np.asarray([3000.0, 180.0, 180.0,  7.0, 0.5,   1.2,  1.2, 0.80,  5.0], dtype='float32')
        
        self.dataFile = "/net/duna/scratch1/aasensio/deepLearning/milne/database/database_6301_hinode_1component.h5"

        f = h5py.File(self.dataFile, 'r')
        pars = f.get("parameters")
        stokes = f.get("stokes")
        self.nModels, _ = pars.shape

        std_values = np.std(np.abs(stokes),axis=0)
        stokes /= std_values[None,:,:]

# Save normalization values        
        np.save('{0}_normalization.npy'.format(self.root), std_values)
                
        self.nTraining = int(self.nModels * 0.9)
        self.nValidation = int(self.nModels * 0.1)
        
        print("Training set: {0}".format(self.nTraining))

        print("Validation set: {0}".format(self.nValidation))

        self.inTrain = []
        for i in range(4):            
            self.inTrain.append(np.atleast_3d(stokes[0:self.nTraining,:,i]).astype('float32'))

        self.inTest = []
        for i in range(4):            
            self.inTest.append(np.atleast_3d(stokes[self.nTraining:,:,i]).astype('float32'))

        self.outTrain = []
        for i in range(9):
            outTrain = np.floor((pars[0:self.nTraining, i] - self.lower[i]) / (self.upper[i] - self.lower[i]) * self.nClasses[i]).astype('int32')            
            self.outTrain.append(np_utils.to_categorical(outTrain, self.nClasses[i]))

        self.outTest = []
        for i in range(9):
            outTest = np.floor((pars[self.nTraining:, i] - self.lower[i]) / (self.upper[i] - self.lower[i]) * self.nClasses[i]).astype('int32')
            self.outTest.append(np_utils.to_categorical(outTest, self.nClasses[i]))

        f.close()
        
        
    def defineNetwork(self):
        print("Setting up network...")

# Stokes I
        sI_input = Input(shape=(self.nLambda,1), name='SI_input')
        x = Convolution1D(nb_filter=self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='valid', init='he_normal', name='convI_1')(sI_input)
        x = Convolution1D(nb_filter=self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='convI_2')(x)
        x = MaxPooling1D(pool_length=self.poolLength, name='poolI_1')(x)
        x = Convolution1D(nb_filter=self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='convI_3')(x)
        x = MaxPooling1D(pool_length=self.poolLength, name='poolI_2')(x)
        x = Convolution1D(nb_filter=self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='convI_4')(x)
        x = MaxPooling1D(pool_length=self.poolLength, name='poolI_3')(x)
        x = Convolution1D(nb_filter=self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='convI_5')(x)
        x = MaxPooling1D(pool_length=self.poolLength, name='poolI_4')(x)
        stI_intermediate = Flatten(name='flatI')(x)

# Stokes Q        
        sQ_input = Input(shape=(self.nLambda,1), name='SQ_input')
        x = Convolution1D(nb_filter=self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='valid', init='he_normal', name='convQ_1')(sQ_input)
        x = Convolution1D(nb_filter=self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='convQ_2')(x)
        x = MaxPooling1D(pool_length=self.poolLength, name='poolQ_1')(x)
        x = Convolution1D(nb_filter=self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='convQ_3')(x)
        x = MaxPooling1D(pool_length=self.poolLength, name='poolQ_2')(x)
        x = Convolution1D(nb_filter=self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='convQ_4')(x)
        x = MaxPooling1D(pool_length=self.poolLength, name='poolQ_3')(x)
        x = Convolution1D(nb_filter=self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='convQ_5')(x)
        x = MaxPooling1D(pool_length=self.poolLength, name='poolQ_4')(x)
        stQ_intermediate = Flatten(name='flatQ')(x)

# Stokes U
        sU_input = Input(shape=(self.nLambda,1), name='SU_input')
        x = Convolution1D(nb_filter=self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='valid', init='he_normal', name='convU_1')(sU_input)
        x = Convolution1D(nb_filter=self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='convU_2')(x)
        x = MaxPooling1D(pool_length=self.poolLength, name='poolU_1')(x)
        x = Convolution1D(nb_filter=self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='convU_3')(x)
        x = MaxPooling1D(pool_length=self.poolLength, name='poolU_2')(x)
        x = Convolution1D(nb_filter=self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='convU_4')(x)
        x = MaxPooling1D(pool_length=self.poolLength, name='poolU_3')(x)
        x = Convolution1D(nb_filter=self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='convU_5')(x)
        x = MaxPooling1D(pool_length=self.poolLength, name='poolU_4')(x)
        stU_intermediate = Flatten(name='flatU')(x)

# Stokes V
        sV_input = Input(shape=(self.nLambda,1), name='SV_input')
        x = Convolution1D(nb_filter=self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='valid', init='he_normal', name='convV_1')(sV_input)
        x = Convolution1D(nb_filter=self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='convV_2')(x)
        x = MaxPooling1D(pool_length=self.poolLength, name='poolV_1')(x)
        x = Convolution1D(nb_filter=self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='convV_3')(x)
        x = MaxPooling1D(pool_length=self.poolLength, name='poolV_2')(x)
        x = Convolution1D(nb_filter=self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='convV_4')(x)
        x = MaxPooling1D(pool_length=self.poolLength, name='poolV_3')(x)
        x = Convolution1D(nb_filter=self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='convV_5')(x)
        x = MaxPooling1D(pool_length=self.poolLength, name='poolV_4')(x)
        stV_intermediate = Flatten(name='flatV')(x)

        x = merge([stI_intermediate, stQ_intermediate, stU_intermediate, stV_intermediate], mode='concat', name='FC')

        x = Dense(400, activation='relu', name='FC2')(x)

        out_BField = Dense(self.nClasses[0], activation='softmax', name='out_BField')(x)
        out_theta = Dense(self.nClasses[1], activation='softmax', name='out_theta')(x)
        out_chi = Dense(self.nClasses[2], activation='softmax', name='out_chi')(x)
        out_vmac = Dense(self.nClasses[3], activation='softmax', name='out_vmac')(x)
        out_a = Dense(self.nClasses[4], activation='softmax', name='out_a')(x)
        out_B0 = Dense(self.nClasses[5], activation='softmax', name='out_B0')(x)
        out_B1 = Dense(self.nClasses[6], activation='softmax', name='out_B1')(x)
        out_doppler = Dense(self.nClasses[7], activation='softmax', name='out_doppler')(x)
        out_kl = Dense(self.nClasses[8], activation='softmax', name='out_kl')(x)

        self.model = Model(input=[sI_input, sQ_input, sU_input, sV_input], output=[out_BField, out_theta, out_chi, out_vmac,
            out_a, out_B0, out_B1, out_doppler, out_kl])
        
        json_string = self.model.to_json()
        f = open('{0}_model.json'.format(self.root), 'w')
        f.write(json_string)
        f.close()

        with open('{0}_summary.txt'.format(self.root), 'w') as f:
            with redirect_stdout(f):
                self.model.summary()

        kerasPlot(self.model, to_file='{0}_model.png'.format(self.root), show_shapes=True)

        
    def compileNetwork(self):        
       # self.model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
        self.model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

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
        
        out = self.model.predict(inTest, batch_size=self.batchSize, verbose=1)

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
