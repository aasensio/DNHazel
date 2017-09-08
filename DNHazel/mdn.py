import matplotlib.pyplot as pl
import numpy as np
import tensorflow as tf
import seaborn as sns
from ipdb import set_trace as stop
import os

os.environ["KERAS_BACKEND"] = "tensorflow"

from keras import backend as K
from keras.layers import Dense, Input, merge
from keras.models import Model
from sklearn.cross_validation import train_test_split

def build_toy_dataset(nsample=40000):
    y_data = np.float32(np.random.uniform(-10.5, 10.5, (1, nsample))).T
    r_data = np.float32(np.random.normal(size=(nsample,1))) # random noise
    x_data = np.float32(np.sin(0.75*y_data)*7.0+y_data*0.5+r_data*1.0)
    return train_test_split(x_data, y_data, random_state=42, train_size=0.1)

def log_sum_exp(x, axis=None):
    """Log-sum-exp trick implementation"""
    x_max = K.max(x, axis=axis, keepdims=True)
    return K.log(K.sum(K.exp(x - x_max), 
                       axis=axis, keepdims=True))+x_max

class MixtureDensityNetwork:
    """
    Mixture density network for outputs y on inputs x.
    p((x,y), (z,theta))
    = sum_{k=1}^K pi_k(x; theta) Normal(y; mu_k(x; theta), sigma_k(x; theta))
    where pi, mu, sigma are the output of a neural network taking x
    as input and with parameters theta. There are no latent variables
    z, which are hidden variables we aim to be Bayesian about.
    """
    def __init__(self, m):
        self.m = m # here K is the amount of Mixtures 
        self.X_train, self.X_test, self.y_train, self.y_test = build_toy_dataset()

    def mapping(self, X):
        """pi, mu, sigma = NN(x; theta)"""
        hidden1 = Dense(15, activation='relu')(X)  # fully-connected layer with 15 hidden units
        hidden2 = Dense(15, activation='relu')(hidden1) 
        self.mus = Dense(self.K)(hidden2) # the means 
        self.sigmas = Dense(self.K, activation=K.exp)(hidden2) # the variance
        self.pi = Dense(self.K, activation=K.softmax)(hidden2) # the mixture components

    def neg_log_normal_mixture(self, true, parameters):        

        means = parameters[:,0*self.m:1*self.m]
        sigmas = parameters[:,1*self.m:2*self.m]
        pi = parameters[:,2*self.m:3*self.m]

        # true_repeated = K.repeat_elements(true, self.m, axis=-1)

        exponent = -0.5 * (true - means)**2 / sigmas
        max_exponent = K.max(exponent, axis=-1, keepdims=True)
        max_exponent_repeated = K.repeat_elements(max_exponent, self.m, axis=-1)

        likelihood = pi * K.exp(exponent - max_exponent_repeated)

        return K.mean(log_sum_exp(likelihood, axis=1))

    def gen_model(self):
        # The network    
        inputs = Input(shape=(1,))
        hidden1 = Dense(15, activation='relu')(inputs)  # fully-connected layer with 15 hidden units
        hidden2 = Dense(15, activation='relu')(hidden1) 

        mus = Dense(self.m)(hidden2) # the means 
        sigmas = Dense(self.m, activation=K.exp)(hidden2) # the variance
        pi = Dense(self.m, activation=K.softmax)(hidden2) # the mixture components

        parameters = merge([mus, sigmas, pi], mode='concat')

        self.model = Model(input=inputs, output=parameters)

        self.model.compile(loss=self.neg_log_normal_mixture, optimizer='adam')

    def fit_model(self):

        self.model.fit(self.X_train, self.y_train,verbose=2)
        

if (__name__ == '__main__'):

    out = MixtureDensityNetwork(5)
    out.gen_model()
    out.fit_model()