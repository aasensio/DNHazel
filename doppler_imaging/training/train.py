import numpy as np
from astropy.io import fits
import platform
import os
import json
import argparse
import h5py
from contextlib import redirect_stdout
import copy
from ipdb import set_trace as stop

if (platform.node() == 'viga'):
    os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32"

os.environ["KERAS_BACKEND"] = "tensorflow"

if (platform.node() != 'viga'):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from keras.callbacks import ModelCheckpoint, Callback, LearningRateScheduler
from keras.models import model_from_json
from keras.optimizers import Adam, RMSprop
from keras.utils import plot_model
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf
import time
import models as nn_model

def flush_file(f):
    f.flush()
    os.fsync(f.fileno())    

class LossHistory(Callback):
    def __init__(self, root, losses, extra, **kwargs):        
        self.losses = losses
        self.losses_batch = copy.deepcopy(losses)
        self.extra = extra

        self.f_epoch = open("/net/vena/scratch/Dropbox/GIT/DeepLearning/losses/{0}_loss.json".format(platform.node()), 'w')
        self.f_epoch.write('['+json.dumps(self.extra))
        
        self.f_epoch_local = open("{0}_loss.json".format(root), 'w')
        self.f_epoch_local.write('['+json.dumps(self.extra))
        
        flush_file(self.f_epoch)
        flush_file(self.f_epoch_local)

    def on_epoch_end(self, batch, logs={}):
        tmp = [time.asctime(),logs.get('loss').tolist(), logs.get('val_loss').tolist(), ktf.get_value(self.model.optimizer.lr).tolist()]
        self.f_epoch.write(','+json.dumps(tmp))
        self.f_epoch_local.write(','+json.dumps(tmp))

        flush_file(self.f_epoch)
        flush_file(self.f_epoch_local)
        
    def on_train_end(self, logs):
        self.f_epoch.write(']')
        self.f_epoch_local.write(']')

        self.f_epoch.close()
        self.f_epoch_local.close()

    def finalize(self):
        pass

class deep_network(object):

    def __init__(self, root, noise, option, lr, lr_multiplier, batch_size, nkernels):

# Only allocate needed memory
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        session = tf.Session(config=config)
        ktf.set_session(session)

        self.normalizations = np.load('normalization.npy')

        self.root = root
        self.option = option
        self.noise = noise
        self.n_filters = nkernels
        self.lr = lr
        self.lr_multiplier = lr_multiplier
        self.batch_size = batch_size

        self.input_training = "/net/viga/scratch1/deepLearning/doppler_imaging/database/training_stars.h5"
        
        f = h5py.File(self.input_training, 'r')
        self.n_training = len(f['modulus'])
        f.close()

        self.input_validation = "/net/viga/scratch1/deepLearning/doppler_imaging/database/validation_stars.h5"
        
        f = h5py.File(self.input_validation, 'r')
        self.n_validation = len(f['modulus'])
        f.close()
        
        self.batchs_per_epoch_training = int(self.n_training / self.batch_size)
        self.batchs_per_epoch_validation = int(self.n_validation / self.batch_size)

        print("Original training set size: {0}".format(self.n_training))
        print("   - Batch size: {0}".format(self.batch_size))
        print("   - Batches per epoch: {0}".format(self.batchs_per_epoch_training))

        print("Original validation set size: {0}".format(self.n_validation))
        print("   - Batch size: {0}".format(self.batch_size))
        print("   - Batches per epoch: {0}".format(self.batchs_per_epoch_validation))

        if (self.n_training % self.batch_size != 0):
            print("ALERT! Something wrong with the batch size")

# Normalization of alpha, beta, gamma
        lmax = 3 
        k = 3.0
        l = np.arange(lmax + 1)
        cl = []
        
        for l in range(lmax+1):
            for i in range(2*l+1):
                cl.append(np.sqrt(1.0 / (1.0+(l/1.0)**k)))

        self.norm_spher_harm = np.array(cl)

    def training_generator(self):
        f = h5py.File(self.input_training, 'r')

        while 1:        
            for i in range(self.batchs_per_epoch_training):

                stokes_train = np.vstack(f['stokesv'][i*self.batch_size:(i+1)*self.batch_size]) / 1e-4
                _, n_lambda = stokes_train.shape
                n_viewpoints = int(n_lambda / 150.0)
                stokes_train = stokes_train.reshape((self.batch_size, n_viewpoints, 150, 1))
                # max = np.max(np.abs(stokes_train), axis=(1,2))
                # stokes_train /= max[:,None,None,:]	

                tmp = np.vstack(f['alpha'][i*self.batch_size:(i+1)*self.batch_size])
                alpha_train = (tmp - self.normalizations[0,:][None,:]) / (self.normalizations[1,:][None,:] - self.normalizations[0,:][None,:]) - 0.5

                # beta_train = np.vstack(f['beta'][i*self.batch_size:(i+1)*self.batch_size]) / self.norm_spher_harm[None,:]
                # gamma_train = np.vstack(f['gamma'][i*self.batch_size:(i+1)*self.batch_size]) / self.norm_spher_harm[None,:]

                # modulus_train = np.hstack(f['modulus'][i*self.batch_size:(i+1)*self.batch_size] / 3e3).astype('float32')

                # yield stokes_train, [modulus_train, alpha_train.astype('float32'), beta_train.astype('float32'), gamma_train.astype('float32')]
                yield stokes_train.astype('float32'), alpha_train[:,0:5].astype('float32')

        f.close()

    def validation_generator(self):

        f = h5py.File(self.input_validation, 'r')
        
        while 1:        
            for i in range(self.batchs_per_epoch_validation):

                stokes_test = np.vstack(f['stokesv'][i*self.batch_size:(i+1)*self.batch_size]) / 1e-4
                _, n_lambda = stokes_test.shape
                n_viewpoints = int(n_lambda / 150.0)
                stokes_test = stokes_test.reshape((self.batch_size, n_viewpoints, 150, 1))
                # stokes_test /= max[:,None,None,:]

                tmp = np.vstack(f['alpha'][i*self.batch_size:(i+1)*self.batch_size])
                alpha_test = (tmp - self.normalizations[0,:][None,:]) / (self.normalizations[1,:][None,:] - self.normalizations[0,:][None,:]) - 0.5

                # beta_test = np.vstack(f['beta'][i*self.batch_size:(i+1)*self.batch_size]) / self.norm_spher_harm[None,:]
                # gamma_test = np.vstack(f['gamma'][i*self.batch_size:(i+1)*self.batch_size]) / self.norm_spher_harm[None,:]


                # modulus_test = np.hstack(f['modulus'][i*self.batch_size:(i+1)*self.batch_size] / 3e3).astype('float32')

                # yield stokes_test, [modulus_test, alpha_test.astype('float32'), beta_test.astype('float32'), gamma_test.astype('float32')]
                yield stokes_test, alpha_test[:,0:5].astype('float32')

        f.close()

    def define_network(self, l2_reg):
        print("Setting up network...")

        self.model = nn_model.zdi(150, self.noise, activation='selu', n_filters=64, l2_reg=1e-7)

        json_string = self.model.to_json()
        f = open('{0}_model.json'.format(self.root), 'w')
        f.write(json_string)
        f.close()

        with open('{0}_summary.txt'.format(self.root), 'w') as f:
            with redirect_stdout(f):
                self.model.summary()

        plot_model(self.model, to_file='{0}_model.png'.format(self.root), show_shapes=True)

    
    def compile_network(self):        
        self.model.compile(loss='mse', optimizer=Adam(lr=self.lr))
        # self.model.compile(loss='mse', optimizer=RMSprop(lr=self.lr, clipvalue=0.5))

    def learning_rate(self, epoch):
        value = self.lr
        if (epoch >= 20):
            value *= self.lr_multiplier
        return value

    def train(self, n_iterations):
        print("Training network...")        
        
        # Recover losses from previous run
        if (self.option == 'continue'):
            with open("{0}_loss.json".format(self.root), 'r') as f:
                losses = json.load(f)
        else:
            losses = []

        self.checkpointer = ModelCheckpoint(filepath="{0}_weights.hdf5".format(self.root), verbose=1, save_best_only=True)
        self.history = LossHistory(self.root, losses, {'name': '{0}'.format(self.root), 'init_t': time.asctime()})

        self.reduce_lr = LearningRateScheduler(self.learning_rate)
        
        self.metrics = self.model.fit_generator(self.training_generator(), self.batchs_per_epoch_training, epochs=n_iterations, 
            callbacks=[self.checkpointer, self.history, self.reduce_lr], validation_data=self.validation_generator(), validation_steps=self.batchs_per_epoch_validation)
        
        self.history.finalize()

if (__name__ == '__main__'):

    parser = argparse.ArgumentParser(description='Train/predict for MFBD')
    parser.add_argument('-o','--output', help='Output files')
    parser.add_argument('-e','--epochs', help='Number of epochs', default=10)
    parser.add_argument('-n','--noise', help='Noise to add during training/prediction', default=0.0)
    parser.add_argument('-k','--kernels', help='N. kernels', default=64)
    parser.add_argument('-a','--action', help='Action', choices=['start', 'continue'], default='start')
    parser.add_argument('-lr','--lr', help='Learning rate', default=1e-3)
    parser.add_argument('-lrm','--lr_multiplier', help='Learning rate multiplier', default=0.96)
    parser.add_argument('-l2','--l2_regularization', help='L2 regularization', default=0.0)
    parser.add_argument('-b','--batchsize', help='Batch size', default=16)
    parsed = vars(parser.parse_args())

    root = parsed['output']
    nEpochs = int(parsed['epochs'])
    option = parsed['action']
    noise = float(parsed['noise'])
    lr = float(parsed['lr'])
    lr_multiplier = float(parsed['lr_multiplier'])
    batch_size = int(parsed['batchsize'])
    nkernels = int(parsed['kernels'])

# Save parameters used
    with open("{0}_args.json".format(root), 'w') as f:
        json.dump(parsed, f)

    out = deep_network(root, noise, option, lr, lr_multiplier, batch_size, nkernels)

    if (option == 'start'):           
        out.define_network(float(parsed['l2_regularization']))        
        
    if (option == 'continue' or option == 'predict'):
        out.read_network()

    if (option == 'start' or option == 'continue'):
        out.compile_network()
        out.train(nEpochs)
