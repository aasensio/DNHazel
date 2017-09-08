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
from keras.utils import np_utils

import models as nn_model

from ipdb import set_trace as stop

def flush_file(f):
    f.flush()
    os.fsync(f.fileno())    

def log_sum_exp(x, axis=None):
    """Log-sum-exp trick implementation"""
    x_max = ktf.max(x, axis=axis, keepdims=True)
    return ktf.log(ktf.sum(ktf.exp(x - x_max), axis=axis, keepdims=True))+x_max

class LossHistory(Callback):
    def __init__(self, root, depth, losses, extra, **kwargs):        
        self.losses = losses
        self.extra = extra

        self.f_epoch = open("/net/vena/scratch/Dropbox/GIT/DeepLearning/losses/{0}_loss.json".format(platform.node()), 'w')
        self.f_epoch.write('['+json.dumps(self.extra))

        self.f_epoch_local = open("{0}_{1}_loss.json".format(root, depth), 'w')
        self.f_epoch_local.write('['+json.dumps(self.extra))

        flush_file(self.f_epoch)
        flush_file(self.f_epoch_local)

    # def on_batch_end(self, batch, logs={}):
    #     tmp = [time.asctime(),logs.get('loss').tolist(), ktf.get_value(self.model.optimizer.lr).tolist()]
    #     self.f_batch.write(','+json.dumps(tmp))
    #     self.f_batch_local.write(','+json.dumps(tmp))

    #     flush_file(self.f_batch)
    #     flush_file(self.f_batch_local)

        
    def on_train_end(self, logs):
        self.f_epoch.write(']')
        self.f_epoch_local.write(']')

        self.f_epoch.close()
        self.f_epoch_local.close()

    def finalize(self):
        pass

class deep_network(object):

    def __init__(self, parsed):

# Only allocate needed memory
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        session = tf.Session(config=config)
        ktf.set_session(session)

        self.root = parsed['output']
        self.batch_size = int(parsed['batchsize'])        
        self.fraction_training = float(parsed['train_fraction'])
        self.noise = float(parsed['noise'])
        self.activation = parsed['activation']
        self.depth = int(parsed['depth'])
        self.n_kernels = int(parsed['kernels'])
        self.lr = float(parsed['lr'])
        self.l2_reg = float(parsed['l2_regularization'])
        self.lr_multiplier = float(parsed['lr_multiplier'])
        self.n_classes = int(parsed['classes'])
        self.c = 9  # Number of variables
        self.method = parsed['method']

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
        self.inTrain.append(self.stokes[:,:,0:self.n_training].T.reshape((self.n_training, self.n_lambda, 4)).astype('float32'))
        self.inTrain.append(self.parameters[-1,0:self.n_training].reshape((self.n_training, 1)).astype('float32'))

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

    def read_data_classification(self):
        print("Reading data...")
        self.f = netcdf.netcdf_file('/net/viga/scratch1/deepLearning/DNHazel/database/database_mus_1000000.db', 'r')
        self.stokes = self.f.variables['stokes'][:]
        self.parameters = self.f.variables['parameters'][:]
        self.n_lambda = len(self.stokes[0,:,0])
        self.n_training = int(self.fraction_training * len(self.stokes[0,0,:]))

        self.std = np.std(self.stokes[:,:,0:self.n_training], axis=-1)
        self.std[0,:] = 1.0

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
        self.inTrain.append((self.stokes[0:1,:,0:self.n_training] / self.std[0:1,:,None]).T.reshape((self.n_training, self.n_lambda, 1)).astype('float32'))
        self.inTrain.append(self.parameters[-1,0:self.n_training].reshape((self.n_training, 1)).astype('float32'))

        self.outTrain = []
        for i in range(4):
            outTrain = np.floor((self.parameters[i,0:self.n_training] - self.lower[i]) / (self.upper[i] - self.lower[i]) * self.n_classes).astype('int32')
            self.outTrain.append(np_utils.to_categorical(outTrain, self.n_classes))

# Add outputs for LOS angles
        # outTrain = np.floor((ThB[0:self.n_training] - self.lower[7]) / (self.upper[7] - self.lower[7]) * self.n_classes).astype('int32')
        # self.outTrain.append(np_utils.to_categorical(outTrain, self.n_classes))

        # outTrain = np.floor((PhiB[0:self.n_training] - 0.001 - self.lower[8]) / (self.upper[8] - self.lower[8]) * self.n_classes).astype('int32')
        # self.outTrain.append(np_utils.to_categorical(outTrain, self.n_classes))

        self.f.close()

        l = 1e-4
        self.noise = 1e-4
        tau = 1.0 / self.noise
        
        self.wd = l**2 / (tau * self.n_training)
        self.dd = 1.0 / (tau * self.n_training)

    def define_network(self):
        
        if (self.method == 'mdn'):
            self.model = nn_model.network(self.n_lambda, self.depth, noise=self.noise, activation=self.activation, n_filters=self.n_kernels, l2_reg=self.l2_reg)
            self.model.compile(loss=self.mean_log_Gaussian_like, optimizer=Adam(lr=self.lr))

        if (self.method == 'dropout'):
            self.model = nn_model.network_dropout(self.n_lambda, self.n_classes, self.depth, noise=self.noise, activation=self.activation, 
                n_filters=self.n_kernels, l2_reg=self.l2_reg, wd=self.wd, dd=self.dd)
            self.model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.lr), metrics=['accuracy'])

        if (self.method == 'nodropout'):
            self.model = nn_model.network_nodropout(self.n_lambda, self.n_classes, self.depth, noise=self.noise, activation=self.activation, 
                n_filters=self.n_kernels, l2_reg=self.l2_reg)
            self.model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.lr), metrics=['accuracy'])

        json_string = self.model.to_json()
        f = open('{0}_model.json'.format(self.root), 'w')
        f.write(json_string)
        f.close()

        plot_model(self.model, to_file='{0}_model.png'.format(self.root), show_shapes=True)
        
    def read_network(self):
        print("Reading previous network...")
                
        f = open('{0}_model.json'.format(self.root), 'r')
        json_string = f.read()
        f.close()

        self.model = model_from_json(json_string)
        self.model.load_weights("{0}_weights.hdf5".format(self.root))        

    def learning_rate(self, epoch):
        value = self.lr
        if (epoch >= 20):
            value *= self.lr_multiplier
        return value
        

    def train(self, n_epochs):
        print("Training network...")    
        losses = []

        self.checkpointer = ModelCheckpoint(filepath="{0}_{1}_weights.hdf5".format(self.root, self.depth), verbose=1, save_best_only=True)
        self.history = LossHistory(self.root, self.depth, losses, {'name': '{0}_{1}'.format(self.root, self.depth), 'init_t': time.asctime()})

        self.reduce_lr = LearningRateScheduler(self.learning_rate)
        
        self.metrics = self.model.fit(x=self.inTrain, y=self.outTrain, batch_size=self.batch_size, 
            epochs=n_epochs, validation_split=0.1, callbacks=[self.checkpointer, self.history, self.reduce_lr])
        
        self.history.finalize()

    def mean_log_Gaussian_like(self, y_true, parameters):
        """Mean Log Gaussian Likelihood distribution
        Note: The 'c' variable is obtained as global variable
        """
        components = ktf.reshape(parameters,[-1, 2*9 + 1, self.n_classes])
        
        mu = components[:, 0:9, :]
        sigma = components[:, 9:18, :]
        alpha = components[:, 18, :]

        alpha = ktf.softmax(ktf.clip(alpha,1e-8,1.))
        
        exponent = ktf.log(alpha) - .5 * float(self.c) * ktf.log(2 * np.pi) \
            - ktf.sum(ktf.log(sigma), axis=1) \
            - ktf.sum((ktf.expand_dims(y_true,2) - mu)**2 / (2*(sigma)**2), axis=1)
        
        log_gauss = log_sum_exp(exponent, axis=1)
        res = - ktf.mean(log_gauss)
        return res        

if (__name__ == '__main__'):

    if (len(sys.argv) == 2):
        f = open(sys.argv[-1], 'r')
        tmp = f.readlines()
        f.close()
        parsed = json.loads(f)

    else:

        parser = argparse.ArgumentParser(description='Train/predict for MFBD')
        parser.add_argument('-o','--output', help='Output files')
        parser.add_argument('-e','--epochs', help='Number of epochs', default=10)
        parser.add_argument('-n','--noise', help='Noise to add during training/prediction', default=0.0)
        parser.add_argument('-d','--depth', help='Depth', default=5)
        parser.add_argument('-k','--kernels', help='N. kernels', default=64)
        parser.add_argument('-a','--action', help='Action', choices=['start', 'continue'], required=True)
        parser.add_argument('-c','--classes', help='N. classes/mixtures', default=8)
        parser.add_argument('-t','--activation', help='Activation', choices=['relu', 'elu'], default='relu')
        parser.add_argument('-m','--method', help='Method', choices=['mdn', 'dropout','nodropout'], default='dropout', required=True)
        parser.add_argument('-f','--train_fraction', help='Train fraction', default=0.9)
        parser.add_argument('-lr','--lr', help='Learning rate', default=1e-4)
        parser.add_argument('-lrm','--lr_multiplier', help='Learning rate multiplier', default=0.96)
        parser.add_argument('-l2','--l2_regularization', help='L2 regularization', default=0.0)
        parser.add_argument('-b','--batchsize', help='Batch size', default=32)
        parsed = vars(parser.parse_args())

    option = parsed['action']
    n_epochs = int(parsed['epochs'])

    out = deep_network(parsed)
    if (parsed['method'] == 'mdn'):
        out.read_data()
    else:
        out.read_data_classification()

    if (option == 'start'):            
        out.define_network()
    elif (option == 'continue'):
        out.readNetwork()
    else:
        print("Option {0} not correct".format(option))
        sys.exit()

    # 
    out.train(n_epochs)