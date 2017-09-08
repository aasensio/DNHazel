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
# from ipdb import set_trace as stop

if (platform.node() == 'viga'):
    os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32"

os.environ["KERAS_BACKEND"] = "tensorflow"

if (platform.node() != 'viga'):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from keras.layers import Input, Dense, Convolution2D, Flatten, merge, MaxPooling2D, UpSampling2D, Cropping2D
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
                output_train = velocity[i*self.batch_size:(i+1)*self.batch_size,:,:,0:2].astype('float32')

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
                output_validation = velocity[i*self.batch_size:(i+1)*self.batch_size,:,:,0:2].astype('float32')

                yield input_validation, output_validation

        f_images.close()
        f_velocity.close()
            
    def defineNetwork(self):
        print("Setting up network...")

        inputs = Input(shape=(self.nx, self.ny, self.n_times))
        conv1a_down = Convolution2D(32, 3, 3, activation='relu', border_mode='valid', init='he_normal')(inputs)
        conv1b_down = Convolution2D(64, 3, 3, activation='relu', border_mode='valid', init='he_normal')(conv1a_down)
        crop1 = Cropping2D(cropping=((40, 40), (40, 40)))(conv1b_down)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1b_down)

        conv2a_down = Convolution2D(64, 3, 3, activation='relu', border_mode='valid', init='he_normal')(pool1)
        conv2b_down = Convolution2D(128, 3, 3, activation='relu', border_mode='valid', init='he_normal')(conv2a_down)
        crop2 = Cropping2D(cropping=((16, 16), (16, 16)))(conv2b_down)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2b_down)

        conv3a_down = Convolution2D(128, 3, 3, activation='relu', border_mode='valid', init='he_normal')(pool2)
        conv3b_down = Convolution2D(256, 3, 3, activation='relu', border_mode='valid', init='he_normal')(conv3a_down)
        crop3 = Cropping2D(cropping=((4, 4), (4, 4)))(conv3b_down)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3b_down)

        conv4a_down = Convolution2D(256, 3, 3, activation='relu', border_mode='valid', init='he_normal')(pool3)
        conv4b_down = Convolution2D(512, 3, 3, activation='relu', border_mode='valid', init='he_normal')(conv4a_down)

        up3 = UpSampling2D(size=(2, 2))(conv4b_down)
        merge3 = merge([up3, crop3], mode='concat', concat_axis=3)

        conv3a_up = Convolution2D(256, 3, 3, activation='relu', border_mode='valid', init='he_normal')(merge3)
        conv3b_up = Convolution2D(256, 3, 3, activation='relu', border_mode='valid', init='he_normal')(conv3a_up)

        up2 = UpSampling2D(size=(2, 2))(conv3b_up)
        merge2 = merge([up2, crop2], mode='concat', concat_axis=3)

        conv2a_up = Convolution2D(128, 3, 3, activation='relu', border_mode='valid', init='he_normal')(merge2)
        conv2b_up = Convolution2D(128, 3, 3, activation='relu', border_mode='valid', init='he_normal')(conv2a_up)

        up1 = UpSampling2D(size=(2, 2))(conv2b_up)
        merge1 = merge([up1, crop1], mode='concat', concat_axis=3)

        conv1a_up = Convolution2D(128, 3, 3, activation='relu', border_mode='valid', init='he_normal')(merge1)
        conv1b_up = Convolution2D(128, 3, 3, activation='relu', border_mode='valid', init='he_normal')(conv1a_up)

        final = Convolution2D(2, 1, 1, activation='linear', border_mode='valid', init='he_normal')(conv1b_up)

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
        self.model.compile(loss='mse', optimizer=Adam(lr=1e-4))
        
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
            callbacks=[self.checkpointer, self.history], validation_data=self.validation_generator(), nb_val_samples=self.n_validation,
            max_q_size=30)
        
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
        out.compileNetwork()
        
    if (option == 'continue' or option == 'predict'):
        out.readNetwork()

    if (option == 'start' or option == 'continue'):
        out.compileNetwork()
        out.trainCNN(nEpochs)
    
    if (option == 'predict'):
        out.predictCNN()