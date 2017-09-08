import numpy as np
import matplotlib.pyplot as pl
import os
import time
import copy
from ipdb import set_trace as stop

os.environ["KERAS_BACKEND"] = "tensorflow"

from keras.layers import Input, Dense, Activation
from keras.models import Model
from keras import callbacks
from keras.callbacks import Callback, ReduceLROnPlateau
import keras.backend as K
import json

class LossHistory(Callback):
    def __init__(self, root, losses, extra):
        self.root = root        
        self.losses = losses
        self.losses_batch = copy.deepcopy(losses)
        self.extra = extra
        self.f_epoch = open("{0}_loss.json".format(self.root), 'w')
        self.f_batch = open("{0}_loss_batch.json".format(self.root), 'w')
        self.f_epoch.write('['+json.dumps(self.extra))
        self.f_batch.write('['+json.dumps(self.extra))

    def on_epoch_end(self, batch, logs={}):        
        tmp = [time.asctime(),logs.get('loss'), logs.get('val_loss'), K.get_value(self.model.optimizer.lr).tolist()]
        self.f_epoch.write(','+json.dumps(tmp))
        # with open("{0}_loss.json".format(self.root), 'w') as f:
            # json.dump([self.extra,self.losses], f)

    def on_batch_end(self, batch, logs={}):
        tmp = [time.asctime(),logs.get('loss').tolist(), K.get_value(self.model.optimizer.lr).tolist()]
        self.f_batch.write(','+json.dumps(tmp))

    def on_train_end(self, logs):
        self.f_batch.write(']')
        self.f_epoch.write(']')
        self.f_batch.close()
        self.f_epoch.close()        

remote = callbacks.RemoteMonitor(root='http://localhost:5000')

np.random.seed(123)

n_data = 1100
n_validation = 20
N = 10
a = np.random.randn(n_data).astype('float32')

noise = 0.3
tau = 1.0 / noise**2
xx = np.linspace(0.0, 1.0, N)

y_train = np.zeros((N,n_data))
y_train = a[None,:] * xx[:,None] + noise * np.random.randn(N,n_data)
y_train = y_train.astype('float32')

inp = Input(shape=(N,), name='input')
x = Dense(25, activation='linear', name='FC')(inp)
x = Dense(1, activation='linear', name='FC2')(x)

model = Model(inputs=inp, outputs=x)

model.compile(loss='mse', optimizer='adam')

history = LossHistory('vena', [], {'name': 'pruebita', 'init_t': time.asctime()})
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-4)

model.fit(y_train.T, np.atleast_2d(a).T, epochs=20, batch_size=30, validation_split=0.1, callbacks=[remote, history, reduce_lr])