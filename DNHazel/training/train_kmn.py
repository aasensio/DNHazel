import numpy as np
import json
import sys
import os
import argparse

from scipy.io import netcdf
from ipdb import set_trace as stop

import keras.backend as K
import tensorflow as tf
from keras.callbacks import CSVLogger, LearningRateScheduler, ModelCheckpoint
from keras.layers import Input, Lambda, Dense, Flatten, BatchNormalization, Activation, Conv1D, add, concatenate, GaussianNoise
from keras.models import Model, load_model
from keras.layers.advanced_activations import PReLU
from keras.optimizers import Adam
from keras.utils import plot_model
from sklearn.cluster import KMeans, AgglomerativeClustering
import pandas as pd
from contextlib import redirect_stdout

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def flush_file(f):
    f.flush()
    os.fsync(f.fileno())    

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


def sample_center_points(y, method='all', k=100, keep_edges=False):
    """
    function to define kernel centers with various downsampling alternatives
    """

    # make sure y is 1D
    y = y.ravel()

    # keep all points as kernel centers
    if method is 'all':
        return y

    # retain outer points to ensure expressiveness at the target borders
    if keep_edges:
        y = np.sort(y)
        centers = np.array([y[0], y[-1]])
        y = y[1:-1]
        # adjust k such that the final output has size k
        k -= 2
    else:
        centers = np.empty(0)

    if method is 'random':
        cluster_centers = np.random.choice(y, k, replace=False)

    # iteratively remove part of pairs that are closest together until everything is at least 'd' apart
    elif method is 'distance':
        raise NotImplementedError

    # use 1-D k-means clustering
    elif method is 'k_means':
        model = KMeans(n_clusters=k, n_jobs=-2)
        model.fit(y.reshape(-1, 1))
        cluster_centers = model.cluster_centers_

    # use agglomerative clustering
    elif method is 'agglomerative':
        model = AgglomerativeClustering(n_clusters=k, linkage='complete')
        model.fit(y.reshape(-1, 1))
        labels = pd.Series(model.labels_, name='label')
        y_s = pd.Series(y, name='y')
        df = pd.concat([y_s, labels], axis=1)
        cluster_centers = df.groupby('label')['y'].mean().values

    else:
        raise ValueError("unknown method '{}'".format(method))

    return np.append(centers, cluster_centers)

class kernel_mixture_network(object):

    def __init__(self, parsed):

# Only allocate needed memory
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        session = tf.Session(config=config)
        K.set_session(session)

        self.root = parsed['model']
        self.action = parsed['action']
        self.batch_size = int(parsed['batch_size'])        
        self.fraction_training = float(parsed['train_fraction'])
        self.lr = float(parsed['lr'])
        self.lr_multiplier = float(parsed['lr_multiplier'])
        self.n_centers = int(parsed['ncenters'])
        self.var = parsed['var']
        self.n_epochs = int(parsed['epochs'])
        self.infer_sigma = parsed['infer_sigma']

        self.lower = np.asarray([0.05, -5.0, 5.0, 0.0, 0.0, 0.0, -180.0, 0.0, -180.0])
        self.upper = np.asarray([3.0, 5.0, 18.0, 0.5, 1000.0, 180.0, 180.0, 180.0, 180.0]) 

        sigmas = parsed['widths']
        
        self.n_sigma = len(sigmas)
        self.sigmas = np.repeat(sigmas, self.n_centers).astype('float32')

        self.oneDivSqrtTwoPI = 1.0 / np.sqrt(2.0*np.pi) # normalisation factor for gaussian.

        with open("{0}_{1}_args.json".format(self.root, self.var), 'w') as f:
            json.dump(parsed, f)


    def gaussian_distribution(self, y, mu, sigma):
        result = (y - mu) / sigma
        result = - 0.5 * (result * result)
        return (K.exp(result) / sigma) * self.oneDivSqrtTwoPI
    
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
        print("Reading data...")
        self.f = netcdf.netcdf_file('/net/viga/scratch1/deepLearning/DNHazel/database/database_mus_1000000.db', 'r')
        self.stokes = self.f.variables['stokes'][:]
        self.parameters = self.f.variables['parameters'][:]
        self.n_lambda = len(self.stokes[0,:,0])
        self.n_training = int(self.fraction_training * len(self.stokes[0,0,:]))

        self.scaling = np.array([1.0,0.01,0.01,0.1])

        mu = self.parameters[7,:]
        thB = self.parameters[5,:] * np.pi / 180.0
        phiB = self.parameters[6,:] * np.pi / 180.0

        cosThB = mu * np.cos(thB) + np.sqrt(1.0-mu**2) * np.sin(thB) * np.cos(phiB)
        sinThB = np.sqrt(1.0 - cosThB**2)

        cosPhiB = (mu * np.sin(thB) * np.cos(phiB) - np.sqrt(1.0-mu**2) * np.cos(thB)) / sinThB
        sinPhiB = np.sin(thB) * np.sin(phiB) / sinThB

        ThB = np.arctan2(sinThB, cosThB) * 180.0 / np.pi
        PhiB = np.arctan2(sinPhiB, cosPhiB) * 180.0 / np.pi

# Add training data, which include the Stokes parameters, the value of the output variable and mu
        self.train = []
        for i in range(4):
            self.train.append((self.stokes[i,:,0:self.n_training] / self.scaling[i]).T.reshape((self.n_training, self.n_lambda, 1)).astype('float32'))
        # self.train.append((self.stokes[:,:,0:self.n_training] / scaling[:,None,None]).T.reshape((self.n_training, self.n_lambda, 4)).astype('float32'))
        if (self.var == 'tau'):
            var = self.parameters[0,0:self.n_training].reshape((self.n_training, 1)) / 2.0
        if (self.var == 'v'):
            var = self.parameters[1,0:self.n_training].reshape((self.n_training, 1)) / 5.0
        if (self.var == 'vth'):
            var = self.parameters[2,0:self.n_training].reshape((self.n_training, 1)) / 10.0
        if (self.var == 'a'):
            var = self.parameters[3,0:self.n_training].reshape((self.n_training, 1)) / 0.5
        if (self.var == 'B'):
            var = self.parameters[4,0:self.n_training].reshape((self.n_training, 1)) / 1000.0
        if (self.var == 'thB'):
            var = thB[0:self.n_training].reshape((self.n_training, 1)) / np.pi
        if (self.var == 'phiB'):
            var = phiB[0:self.n_training].reshape((self.n_training, 1)) / np.pi
        if (self.var == 'thB_LOS'):
            var = ThB[0:self.n_training].reshape((self.n_training, 1)) / np.pi
        if (self.var == 'phiN_LOS'):
            var = PhiB[0:self.n_training].reshape((self.n_training, 1)) / np.pi

        self.train.append(var.astype('float32'))
        self.train.append(self.parameters[-1,0:self.n_training].reshape((self.n_training, 1)).astype('float32'))

# Find the k-means centers of the clusters
        print("Computing k-means...")
        self.center_locs = sample_center_points(var, method='k_means', 
                                                k=self.n_centers, keep_edges=True).astype('float32')

        self.center_locs = np.tile(self.center_locs, self.n_sigma)

        np.savez("{0}_{1}_centers.npz".format(self.root, self.var), center_locs=self.center_locs, sigmas=self.sigmas)
        
    def build_estimator(self):

# Inputs
        input_I = Input(shape=(self.n_lambda,1), name='stokes_I')
        input_I_noise = GaussianNoise(stddev=1e-4 / self.scaling[0])(input_I)

        input_Q = Input(shape=(self.n_lambda,1), name='stokes_Q')
        input_Q_noise = GaussianNoise(stddev=1e-4 / self.scaling[1])(input_Q)

        input_U = Input(shape=(self.n_lambda,1), name='stokes_U')
        input_U_noise = GaussianNoise(stddev=1e-4 / self.scaling[2])(input_U)

        input_V = Input(shape=(self.n_lambda,1), name='stokes_V')
        input_V_noise = GaussianNoise(stddev=1e-4 / self.scaling[3])(input_V)        

        y_true = Input(shape=(1,), name='y_true')
        mu_input = Input(shape=(1,), name='mu_input')

        input_x = concatenate([input_I_noise,input_Q_noise,input_U_noise,input_V_noise])

        kernels = [64, 64, 64]

# Neural network
        x = Conv1D(64, 3, padding='same', kernel_initializer='he_normal', name='conv_1')(input_x)
        x = PReLU()(x)

        for i in range(3):
            x = residual(x, kernels[i], 'prelu', strides=2)
    
        intermediate = Flatten(name='flat')(x)
        intermediate_conv = concatenate([intermediate, mu_input], name='FC')

# Output weights
        weights = Dense(self.n_centers*self.n_sigma, activation='softmax', name='weights')(intermediate_conv)
        if (self.infer_sigma == 'yes'):
            sigma_global = Dense(1, activation='softplus', name='sigma')(intermediate_conv)

# Definition of the loss function
            loss = Lambda(self.mdn_loss_function_infer, output_shape=(1,), name='loss')([y_true, weights, sigma_global])

        else:
# Definition of the loss function
            loss = Lambda(self.mdn_loss_function, output_shape=(1,), name='loss')([y_true, weights])

        self.model = Model(inputs=[input_I,input_Q,input_U,input_V, y_true, mu_input], outputs=[loss])
        self.model.add_loss(loss)
    
# Compile with the loss weight set to None, so it will be omitted
        self.model.compile(loss=[None], loss_weights=[None], optimizer=Adam(lr=self.lr))

        with open('{0}_{1}_summary.txt'.format(self.root, self.var), 'w') as f:
            with redirect_stdout(f):
                self.model.summary()

        plot_model(self.model, to_file='{0}_{1}_model.png'.format(self.root, self.var), show_shapes=True)

        if (self.action == 'continue'):
            self.model.load_weights("{0}_{1}_model.h5".format(self.root, self.var))

# Now generate a second network that ends up in the weights for later evaluation        
        if (self.infer_sigma == 'yes'):
            self.model_weights = Model(inputs=self.model.input,
                                 outputs=[self.model.get_layer('weights').output, self.model.get_layer('sigma').output])
        else:
            self.model_weights = Model(inputs=self.model.input,
                                 outputs=self.model.get_layer('weights').output)

    def train_network(self):
        print("Compiling network...")
        
        self.build_estimator()
        
        if (self.action == 'continue'):
            append = True
        else:
            append = False
            
        self.csv = CSVLogger("{0}_{1}_loss.csv".format(self.root, self.var), append=append)

        self.checkpointer = ModelCheckpoint(filepath="{0}_{1}_best.h5".format(self.root, self.var), verbose=1, save_best_only=True)

        self.reduce_lr = LearningRateScheduler(self.learning_rate)

        self.model.fit(x=self.train, epochs=self.n_epochs, batch_size=self.batch_size, validation_split=0.1,
            shuffle=True, callbacks=[self.csv, self.checkpointer, self.reduce_lr])

        self.model.save("{0}_{1}_model.h5".format(self.root, self.var))

    def learning_rate(self, epoch):
        value = self.lr
        if (epoch >= 20):
            value *= self.lr_multiplier
        return value


if (__name__ == '__main__'):

    if (len(sys.argv) == 1):
        print()

    if (len(sys.argv) == 2):
        f = open(sys.argv[-1], 'r')
        tmp = f.read()
        f.close()
        parsed = json.loads(tmp)

    else:

        parser = argparse.ArgumentParser(description='Train/predict for MFBD')
        parser.add_argument('-o','--model', help='Model name', required=True)
        parser.add_argument('-e','--epochs', help='Number of epochs', default=10)
        parser.add_argument('-a','--action', help='Action', choices=['start', 'continue'], default='start')
        parser.add_argument('-i','--infer_sigma', help='Infer sigma', choices=['yes', 'no'], default='no')
        parser.add_argument('-c','--ncenters', help='N. centers', default=50)
        parser.add_argument('-f','--train_fraction', help='Train fraction', default=0.9)
        parser.add_argument('-lr','--lr', help='Learning rate', default=1e-4)
        parser.add_argument('-lrm','--lr_multiplier', help='Learning rate multiplier', default=1.0)
        parser.add_argument('-b','--batch_size', help='Batch size', default=256)
        parser.add_argument('-w','--widths', help='Widths of kernels', nargs='+', type=float, required=True)
        parser.add_argument('-v','--var', help='Variable to train', choices=['tau','v','vth','a','B','thB','phiB','thB_LOS','phiB_LOS'], 
            default='tau', required=True)

        parsed = vars(parser.parse_args())

    out = kernel_mixture_network(parsed)
    out.read_data()
    
    out.train_network()
