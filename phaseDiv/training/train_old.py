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

from keras.layers import Input, Conv2D, Activation, BatchNormalization, GaussianNoise, add, UpSampling2D
from keras.callbacks import ModelCheckpoint, Callback, ReduceLROnPlateau
from keras.models import Model, model_from_json
from keras.optimizers import Adam
from keras.utils import plot_model
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf
import time
import encdec_model as nn_model

class LossHistory(Callback):
    def __init__(self, root, losses, extra):
        self.root = root        
        self.losses = losses
        self.extra = extra

    def on_epoch_end(self, batch, logs={}):
        self.losses.append([time.asctime(),logs.get('loss'), logs.get('val_loss'), ktf.get_value(self.model.optimizer.lr).tolist()])
        with open("{0}_loss.json".format(self.root), 'w') as f:
            json.dump([self.extra, self.losses], f)

        with open("/home/aasensio/{0}_loss.json".format(platform.node()), 'w') as f:
            json.dump([self.extra, self.losses], f)

    def finalize(self):
        pass

class phasediv(object):

    def __init__(self, root, noise, option, depth):

# Only allocate needed memory
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        session = tf.Session(config=config)
        ktf.set_session(session)


        self.root = root
        self.option = option
        self.noise = noise
        self.depth = depth

        self.batch_size = 128

        self.input_file_images_training = "/net/duna/scratch1/aasensio/deepLearning/phaseDiv/database/database_images.h5"
        self.input_file_images_validation = "/net/duna/scratch1/aasensio/deepLearning/phaseDiv/database/database_images_validation.h5"
        
        f = h5py.File(self.input_file_images_training, 'r')
        self.n_training_orig, self.nx, self.ny, _ = f.get("intensity").shape
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

        while 1:        
            for i in range(self.batchs_per_epoch_training):

                input_train = images[i*self.batch_size:(i+1)*self.batch_size,:,:,1:3].astype('float32')
                output_train = images[i*self.batch_size:(i+1)*self.batch_size,:,:,0:1].astype('float32')

                yield input_train, output_train

        f_images.close()
        
    def validation_generator(self):
        f_images = h5py.File(self.input_file_images_validation, 'r')
        images = f_images.get("intensity")        
        
        while 1:        
            for i in range(self.batchs_per_epoch_validation):

                input_validation = images[i*self.batch_size:(i+1)*self.batch_size,:,:,1:3].astype('float32')
                output_validation = images[i*self.batch_size:(i+1)*self.batch_size,:,:,0:1].astype('float32')

                yield input_validation, output_validation

        f_images.close()    

    def define_network(self):
        print("Setting up network...")

        self.model = nn_model.define_network(self.nx, self.ny, self.noise, self.depth)
            
        json_string = self.model.to_json()
        f = open('{0}_{1}_model.json'.format(self.root, self.depth), 'w')
        f.write(json_string)
        f.close()

        with open('{0}_{1}_summary.txt'.format(self.root, self.depth), 'w') as f:
            with redirect_stdout(f):
                self.model.summary()

        plot_model(self.model, to_file='{0}_{1}_model.png'.format(self.root, self.depth), show_shapes=True)
        
    def compile_network(self):        
        self.model.compile(loss='mse', optimizer=Adam(lr=5e-4))
        
    def read_network(self):
        print("Reading previous network...")
                
        f = open('{0}_{1}_model.json'.format(self.root, self.depth), 'r')
        json_string = f.read()
        f.close()

        self.model = model_from_json(json_string)
        self.model.load_weights("{0}_{1}_weights.hdf5".format(self.root, self.depth))

    def train(self, n_iterations):
        print("Training network...")        
        
        # Recover losses from previous run
        if (self.option == 'continue'):
            with open("{0}_{1}_loss.json".format(self.root, self.depth), 'r') as f:
                losses = json.load(f)
        else:
            losses = []

        self.checkpointer = ModelCheckpoint(filepath="{0}_{1}_weights.hdf5".format(self.root, self.depth), verbose=1, save_best_only=True)
        self.history = LossHistory(self.root, losses, {'name': '{0}_{1}'.format(self.root, self.depth), 'init_t': time.asctime()})
        self.reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-4)
        
        self.metrics = self.model.fit_generator(self.training_generator(), self.batchs_per_epoch_training, epochs=n_iterations, 
            callbacks=[self.checkpointer, self.history], validation_data=self.validation_generator(), validation_steps=self.batchs_per_epoch_validation)
        
        self.history.finalize()

if (__name__ == '__main__'):

    parser = argparse.ArgumentParser(description='Train/predict for phase diversity')
    parser.add_argument('-o','--out', help='Output files')
    parser.add_argument('-e','--epochs', help='Number of epochs', default=10)
    parser.add_argument('-n','--noise', help='Noise to add during training/prediction', default=0.0)
    parser.add_argument('-d','--depth', help='Depth', default=5)
    parser.add_argument('-a','--action', help='Action', choices=['start', 'continue', 'predict'], required=True)
    parsed = vars(parser.parse_args())

    root = parsed['out']
    nEpochs = int(parsed['epochs'])
    option = parsed['action']
    noise = float(parsed['noise'])
    depth = int(parsed['depth'])

    out = phasediv(root, noise, option, depth)

    if (option == 'start'):           
        out.define_network()        
        
    if (option == 'continue' or option == 'predict'):
        out.read_network()

    if (option == 'start' or option == 'continue'):
        out.compile_network()
        out.train(nEpochs)