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

from keras.layers import Input, Dense, Convolution2D, Flatten, merge, Deconvolution2D, Activation
from keras.callbacks import ModelCheckpoint, Callback
from keras.models import Model, model_from_json
from keras.optimizers import Adam
from keras.utils.visualize_util import plot as kerasPlot

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
        self.option = option

        self.n_filters = 64
        self.kernel_size = 3        
        self.batch_size = 32
        self.n_conv_layers = 10
        self.stride = 1
        self.skip_frequency = 2

        self.input_file_images_training = "/scratch1/aasensio/deepLearning/opticalFlow/database/database_images.h5"
        self.input_file_velocity_training = "/scratch1/aasensio/deepLearning/opticalFlow/database/database_velocity.h5"

        self.input_file_images_validation = "/scratch1/aasensio/deepLearning/opticalFlow/database/database_images_validation.h5"
        self.input_file_velocity_validation = "/scratch1/aasensio/deepLearning/opticalFlow/database/database_velocity_validation.h5"

        f = h5py.File(self.input_file_images_training, 'r')
        self.n_training_orig, self.nx, self.ny, self.n_times = f.get("intensity").shape        
        f.close()

        f = h5py.File(self.input_file_images_validation, 'r')
        self.n_validation_orig, _, _, _ = f.get("intensity").shape        
        f.close()
        
        self.batchs_per_epoch_training = int(self.n_training_orig / self.batch_size)
        self.batchs_per_epoch_validation = int(self.n_validation_orig / self.batch_size)

        self.n_training = self.batchs_per_epoch_training * self.batch_size
        self.n_validation = self.batchs_per_epoch_validation * self.batch_size

        print("Original training set size: {0}".format(self.n_training_orig))
        print("   - Final training set size: {0}".format(self.n_training))
        print("   - Batch size: {0}".format(self.batch_size))
        print("   - Batches per epoch: {0}".format(self.batchs_per_epoch_training))

        print("Original validation set size: {0}".format(self.n_validation_orig))
        print("   - Final validation set size: {0}".format(self.n_validation))
        print("   - Batch size: {0}".format(self.batch_size))
        print("   - Batches per epoch: {0}".format(self.batchs_per_epoch_validation))

    def training_generator(self):
        f_images = h5py.File(self.input_file_images_training, 'r')
        images = f_images.get("intensity")

        f_velocity = h5py.File(self.input_file_velocity_training, 'r')
        velocity = f_velocity.get("velocity")

        while 1:        
            for i in range(self.batchs_per_epoch_training):

                input_train = images[i*self.batch_size:(i+1)*self.batch_size,:,:,:].astype('float32')
                output_train = velocity[i*self.batch_size:(i+1)*self.batch_size,:,:,:].astype('float32')

                yield input_train, output_train

        f_images.close()
        f_velocity.close()

    def validation_generator(self):
        f_images = h5py.File(self.input_file_images_validation, 'r')
        images = f_images.get("intensity")

        f_velocity = h5py.File(self.input_file_velocity_validation, 'r')
        velocity = f_velocity.get("velocity")
        
        while 1:        
            for i in range(self.batchs_per_epoch_validation):

                input_validation = images[i*self.batch_size:(i+1)*self.batch_size,:,:,:].astype('float32')
                output_validation = velocity[i*self.batch_size:(i+1)*self.batch_size,:,:,:].astype('float32')

                yield input_validation, output_validation

        f_images.close()
        f_velocity.close()
            
    def defineNetwork(self):
        print("Setting up network...")

        conv = [None] * self.n_conv_layers
        deconv = [None] * self.n_conv_layers

        inputs = Input(shape=(self.nx, self.ny, self.n_times))
        conv[0] = Convolution2D(self.n_filters, 3, 3, activation='relu', subsample=(self.stride,self.stride), border_mode='same', init='he_normal')(inputs)
        for i in range(self.n_conv_layers-1):
            conv[i+1] = Convolution2D(self.n_filters, 3, 3, activation='relu', subsample=(self.stride,self.stride), border_mode='same', init='he_normal')(conv[i])

        deconv[0] = Deconvolution2D(self.n_filters, 3, 3, activation='relu', output_shape=(self.batch_size, self.nx, self.ny,self.n_filters), subsample=(self.stride,self.stride), border_mode='same', init='he_normal')(conv[-1])
        for i in range(self.n_conv_layers-1):
            if (i % self.skip_frequency == 0):
                x = Deconvolution2D(self.n_filters, 3, 3, output_shape=(self.batch_size,self.nx, self.ny,self.n_filters), activation='relu', subsample=(self.stride,self.stride), border_mode='same', init='he_normal')(deconv[i])                
                x = merge([conv[self.n_conv_layers-i-2], x], mode='sum')
                deconv[i+1] = Activation('relu')(x)

            else:
                deconv[i+1] = Deconvolution2D(self.n_filters, 3, 3, output_shape=(self.batch_size,self.nx, self.ny,self.n_filters), activation='relu', subsample=(self.stride,self.stride), border_mode='same', init='he_normal')(deconv[i])

        final = Deconvolution2D(6, 1, 1, output_shape=(self.batch_size,self.nx, self.ny, 6), activation='linear', subsample=(self.stride,self.stride), border_mode='same', init='he_normal')(deconv[-1])

        self.model = Model(input=inputs, output=final)
                
        json_string = self.model.to_json()
        f = open('{0}_model.json'.format(self.root), 'w')
        f.write(json_string)
        f.close()

        with open('{0}_summary.txt'.format(self.root), 'w') as f:
            with redirect_stdout(f):
                self.model.summary()

        kerasPlot(self.model, to_file='{0}_model.png'.format(self.root), show_shapes=True)

    def compileNetwork(self):        
        self.model.compile(loss='mse', optimizer='adam')
        
    def readNetwork(self):
        print("Reading previous network...")
                
        f = open('{0}_model.json'.format(self.root), 'r')
        json_string = f.read()
        f.close()

        self.model = model_from_json(json_string)
        self.model.load_weights("{0}_weights.hdf5".format(self.root))

    def trainCNN(self, n_iterations):
        print("Training network...")        
        
        # Recover losses from previous run
        if (self.option == 'continue'):
            with open("{0}_loss.json".format(self.root), 'r') as f:
                losses = json.load(f)
        else:
            losses = []

        self.checkpointer = ModelCheckpoint(filepath="{0}_weights.hdf5".format(self.root), verbose=1, save_best_only=True)
        self.history = LossHistory(self.root, losses)
        
        self.metrics = self.model.fit_generator(self.training_generator(), self.n_training, nb_epoch=n_iterations, 
            callbacks=[self.checkpointer, self.history], validation_data=self.validation_generator(), nb_val_samples=self.n_validation)
        
        self.history.finalize()

    def predictCNN(self):
        print("Predicting validation data...")        
        f_images = h5py.File(self.input_file_images_validation, 'r')
        images = f_images.get("intensity")

        input_validation = images[0:5,:,:,:]

        out = self.model.predict(input_validation, verbose=1)

        print("Saving validation data...")
        with open("{0}_pred.pkl".format(self.root), "wb") as outfile:
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
    noise = parsed['noise']

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