import numpy as np
import platform
import os
import simplejson
import sys

if (platform.node() == 'viga'):
    os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32"
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Convolution1D, MaxPooling1D, Flatten
from keras.callbacks import ModelCheckpoint, Callback
from keras.models import Model, model_from_json
from keras.utils.visualize_util import plot as kerasPlot
import keras.optimizers
# from keras.optimizers import Nadam
from keras.utils import np_utils
from scipy.io import netcdf
from ipdb import set_trace as stop

class LossHistory(Callback):
    def __init__(self, root):
        self.root = root
        self.fHistory = open("{0}_loss.json".format(self.root), 'w')

    def on_train_begin(self, logs={}):        
        self.losses = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append([logs.get('loss'), logs.get('val_loss'), logs.get('acc'), logs.get('val_acc')])
        simplejson.dump([logs.get('loss'), logs.get('val_loss'), logs.get('acc'), logs.get('val_acc')], self.fHistory)

    def finalize(self):
        self.fHistory.close()

class trainDNNFull(object):

    def __init__(self, root):
        self.root = root
        self.nFeatures = 50
        self.kernelSize = 5
        self.poolLength = 2
        self.nLambda = 128
        self.batchSize = 128
        self.nImages = 10
        self.nTrainSamples = 90000

        self.lower = np.asarray([0.05, -5.0, 5.0, 0.0, 0.0, 0.0, -180.0])
        self.upper = np.asarray([3.0, 5.0, 18.0, 0.5, 1000.0, 180.0, 180.0])        

    def readData(self):
        print("Reading data...")
        self.f = netcdf.netcdf_file('../database/database.db', 'r')
        self.stokes = self.f.variables['stokes'][:]
        self.parameters = self.f.variables['parameters'][:]

        self.inTrain = self.stokes[0,:,0:self.nTrainSamples].T

        self.inTrain = self.inTrain.reshape((self.nTrainSamples, self.nLambda, 1))
        
        self.outTrain = []
        for i in range(4):
            outTrain = np.floor((self.parameters[i,0:self.nTrainSamples] - self.lower[i]) / (self.upper[i] - self.lower[i]) * 20).astype('int32')
            self.outTrain.append(np_utils.to_categorical(outTrain, 20))

        self.f.close()
        
    def defineNetwork(self):
        print("Setting up network...")

        sI_input = Input(shape=(self.nLambda,1))
        x = Convolution1D(nb_filter=self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='same', init='he_normal')(sI_input)
        x = MaxPooling1D(pool_length=self.poolLength)(x)

        x = Convolution1D(nb_filter=self.nFeatures, filter_length=2*self.kernelSize, activation='relu', border_mode='same', init='he_normal')(x)
        x = MaxPooling1D(pool_length=self.poolLength)(x)

        x = Convolution1D(nb_filter=self.nFeatures, filter_length=3*self.kernelSize, activation='relu', border_mode='same', init='he_normal')(x)
        x = MaxPooling1D(pool_length=self.poolLength)(x)

        x = Convolution1D(nb_filter=self.nFeatures, filter_length=4*self.kernelSize, activation='relu', border_mode='same', init='he_normal')(x)
        x = MaxPooling1D(pool_length=self.poolLength)(x)

        x = Convolution1D(nb_filter=self.nFeatures, filter_length=5*self.kernelSize, activation='relu', border_mode='same', init='he_normal')(x)
        x = MaxPooling1D(pool_length=self.poolLength)(x)

        x = Flatten()(x)

        # x_tau = Dense(400, activation='relu')(x)
        out_tau = Dense(20, activation='softmax')(x)

        # x_v = Dense(400, activation='relu')(x)
        out_v = Dense(20, activation='softmax')(x)

        # x_vth = Dense(400, activation='relu')(x)
        out_vth = Dense(20, activation='softmax')(x)

        # x_a = Dense(400, activation='relu')(x)
        out_a = Dense(20, activation='softmax')(x)

        self.model = Model(input=sI_input, output=[out_tau, out_v, out_vth, out_a])

        json_string = self.model.to_json()
        f = open('{0}_model.json'.format(self.root), 'w')
        f.write(json_string)
        f.close()

        kerasPlot(self.model, to_file='{0}model.png'.format(self.root), show_shapes=True)
            
        self.model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])

    def readNetwork(self):
        print("Reading previous network...")
                
        f = open('{0}_model.json'.format(self.root), 'r')
        json_string = f.read()
        f.close()

        self.model = model_from_json(json_string)
        self.model.load_weights("{0}_weights.hdf5".format(self.root))
        
        adam = keras.optimizers.adam(lr=1e-4)
        self.model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])


    def trainCNN(self, nIterations):
        print("Training network...")        
        self.checkpointer = ModelCheckpoint(filepath="{0}_weights.hdf5".format(self.root), verbose=1, save_best_only=True)
        self.history = LossHistory(self.root)
        self.metrics = self.model.fit(self.inTrain, self.outTrain, batch_size=self.batchSize, 
            nb_epoch=nIterations, validation_split=0.2, shuffle=False, callbacks=[self.checkpointer, self.history])
        self.history.finalize()
        
        # self.model.fit(self.XTrainSet, self.YTrainSet, batch_size=self.batchSize, nb_epoch=self.nbEpoch, validation_split=0.2)


if (__name__ == '__main__'):
    if (len(sys.argv) != 4):
        print("Usage:")
        print("python trainStI.py root_file nEpochs start/continue")
        print("Example:")
        print("python trainStI.py cnns/I-tau_vth_v_a 20 start")
        sys.exit()
    else:
        root = sys.argv[1]
        nEpochs = int(sys.argv[2])
        option = sys.argv[3]
        out = trainDNNFull(root)
        out.readData()
        if (option == 'start'):            
            out.defineNetwork()
        elif (option == 'continue'):
            out.readNetwork()
        else:
            print("Option {0} not correct".format(option))
            sys.exit()
        out.trainCNN(nEpochs)