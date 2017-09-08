import numpy as np
import matplotlib.pyplot as pl
import os
from ipdb import set_trace as stop

os.environ["KERAS_BACKEND"] = "tensorflow"

from keras.optimizers import Adam
from keras.layers import Dense, LSTM, Input, TimeDistributed, Flatten
from keras.models import Model
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf
from keras.utils import plot_model


class deep_lstm(object):

    def __init__(self):

# Only allocate needed memory
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        session = tf.Session(config=config)
        ktf.set_session(session)

        self.batch_size = 16

        self.x_train = []
        self.y_train = []
        for i in range(1000):
            n = np.random.randint(3, high=10)
            x_train = np.zeros((self.batch_size, n, 2, 1))
            x_train[:,:,:,0] = np.random.rand(self.batch_size, n, 2)            
            a = np.random.rand(self.batch_size)
            y_train = a[:,None,None,None] * x_train
            self.x_train.append(y_train)
            self.y_train.append(a)

        self.max = np.max(np.array(self.y_train))
        self.min = np.min(np.array(self.y_train))

        for i in range(1000):
            self.x_train[i] = (self.x_train[i] - self.min) / (self.max - self.min)

    def define_network(self):

        st = Input(shape=(None, 2, 1), name='input')
        x = TimeDistributed(Flatten(), name='flatten')(st)

        x = LSTM(64)(x)        
        output_alpha = Dense(1, name='alpha')(x)

        self.model = Model(inputs=st, outputs=output_alpha)

        plot_model(self.model, to_file='lstm_model.png', show_shapes=True)

    def training_generator(self):
        
        while 1:        
            for i in range(1000):            
                yield self.x_train[i].astype('float32'), self.y_train[i].astype('float32')
        
    def compile_network(self):        
        self.model.compile(loss='mse', optimizer=Adam(lr=1e-3))

    def train(self, n_iterations):
        print("Training network...")        
          
        self.metrics = self.model.fit_generator(self.training_generator(), 1000, epochs=n_iterations)

    def test(self):
        n = np.array([3,5,7,10])
        out_syn = np.zeros((4,16))
        out_nn = np.zeros((4,16))

        for i in range(4):            
            x_train = np.zeros((self.batch_size, n[i], 2, 1))
            x_train[:,:,:,0] = np.random.rand(self.batch_size, n[i], 2)
            a = np.random.rand(self.batch_size)
            y_train = a[:,None,None,None] * x_train

            y_train = (y_train - self.min) / (self.max - self.min)

            pred = self.model.predict(y_train.astype('float32'), batch_size=16)

            out_syn[i,:] = a
            out_nn[i,:] = pred.flatten()

        f, ax = pl.subplots(nrows=2, ncols=2)
        ax = ax.flatten()
        for i in range(4):
            ax[i].plot(out_syn[i,:], out_nn[i,:], '.')
            ax[i].plot([0,1], [0,1])

        pl.show()

        return out_nn, out_syn
                
if (__name__ == '__main__'):

    out = deep_lstm()

    out.define_network()
        
    out.compile_network()

    out.train(2)

    nn, syn = out.test()