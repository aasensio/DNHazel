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

os.environ["KERAS_BACKEND"] = "tensorflow"

if (platform.node() != 'viga'):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from keras.layers import Input, Convolution2D, add, Activation, BatchNormalization
from keras.callbacks import ModelCheckpoint, Callback
from keras.models import Model, model_from_json
from keras.optimizers import Adam
from keras.utils import plot_model
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf

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

class resnet(object):

    def __init__(self, root, noise):

# Only allocate needed memory
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        session = tf.Session(config=config)
        ktf.set_session(session)


        self.root = root

        self.n_filters = 64
        self.kernel_size = 3      
        self.batch_size = 32
        self.n_conv_layers = 10        

        self.input_file_images_validation = "/net/duna/scratch1/aasensio/deepLearning/stars/database/database_validation.h5"
                
        f = h5py.File(self.input_file_images_validation, 'r')
        self.n_validation_orig, self.nx, self.ny, self.n_images = f.get("image").shape        
        f.close()
        
        self.batchs_per_epoch_validation = int(self.n_validation_orig / self.batch_size)

        self.n_validation = self.batchs_per_epoch_validation * self.batch_size        

        print("Original validation set size: {0}".format(self.n_validation_orig))
        print("   - Final validation set size: {0}".format(self.n_validation))
        print("   - Batch size: {0}".format(self.batch_size))
        print("   - Batches per epoch: {0}".format(self.batchs_per_epoch_validation))

    def validation_generator(self):
        f_images = h5py.File(self.input_file_images_validation, 'r')
        images = f_images.get("image")
    
        while 1:        
            for i in range(self.batchs_per_epoch_validation):

                input_validation = images[i*self.batch_size:(i+1)*self.batch_size,:,:,0:1].astype('float32')
                output_validation = images[i*self.batch_size:(i+1)*self.batch_size,:,:,1:2].astype('float32')

                yield input_validation, output_validation

        f_images.close()

    def residual(self, inputs):
        x = Convolution2D(self.n_filters, (3, 3), padding='same', kernel_initializer='he_normal')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Convolution2D(self.n_filters, (3, 3), padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = add([x, inputs])

        return x
            
    def defineNetwork(self):
        print("Setting up network...")

        inputs = Input(shape=(self.nx, self.ny, 1))
        conv = Convolution2D(self.n_filters, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)

        x = self.residual(conv)
        for i in range(self.n_conv_layers):
            x = self.residual(x)

        x = Convolution2D(self.n_filters, (3, 3), padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = add([x, conv])

        final = Convolution2D(1, (1, 1), activation='linear', padding='same', kernel_initializer='he_normal')(x)

        self.model = Model(inputs=inputs, outputs=final)
                
        json_string = self.model.to_json()
        f = open('{0}_model.json'.format(self.root), 'w')
        f.write(json_string)
        f.close()

        with open('{0}_summary.txt'.format(self.root), 'w') as f:
            with redirect_stdout(f):
                self.model.summary()

    def compileNetwork(self):        
        self.model.compile(loss='mse', optimizer=Adam(lr=1e-4))
        
    def readNetwork(self):
        print("Reading previous network...")
                
        f = open('{0}_model.json'.format(self.root), 'r')
        json_string = f.read()
        f.close()

        self.model = model_from_json(json_string)
        self.model.load_weights("{0}_weights.hdf5".format(self.root))

    def predict(self):            
        out = self.model.predict_generator(self.validation_generator(), self.batchs_per_epoch_validation)
        stop()
        


if (__name__ == '__main__'):    

    out = resnet('../training/networks/resnet', 1e-3)
    out.readNetwork()
    out.compileNetwork()