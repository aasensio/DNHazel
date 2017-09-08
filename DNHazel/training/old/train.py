import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Convolution1D, MaxPooling1D, Flatten
from keras.callbacks import ModelCheckpoint, Callback
from keras.optimizers import Nadam
from keras.utils import np_utils
from scipy.io import netcdf
# from ipdb import set_trace as stop

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append([logs.get('loss'), logs.get('val_loss'), logs.get('acc'), logs.get('val_acc')])

class trainDNN(object):

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

        outTrain = np.floor((self.parameters[0,0:self.nTrainSamples] - self.lower[0]) / (self.upper[0] - self.lower[0]) * 20).astype('int32')
        self.outTrain = np_utils.to_categorical(outTrain, 20)
        
    def defineNetwork(self):
        print("Setting up network...")
        self.model = Sequential()

        self.model.add(Convolution1D(nb_filter=self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='same', init='he_normal', 
            input_shape=(self.nLambda,1)))
        self.model.add(MaxPooling1D(pool_length=self.poolLength))

        self.model.add(Convolution1D(nb_filter=2*self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='same', init='he_normal'))
        self.model.add(MaxPooling1D(pool_length=self.poolLength))

        self.model.add(Convolution1D(nb_filter=3*self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='same', init='he_normal'))
        self.model.add(MaxPooling1D(pool_length=self.poolLength))

        self.model.add(Convolution1D(nb_filter=4*self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='same', init='he_normal'))
        self.model.add(MaxPooling1D(pool_length=self.poolLength))

        self.model.add(Convolution1D(nb_filter=5*self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='same', init='he_normal'))
        self.model.add(MaxPooling1D(pool_length=self.poolLength))

        self.model.add(Convolution1D(nb_filter=6*self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='same', init='he_normal'))
        self.model.add(MaxPooling1D(pool_length=self.poolLength))

        self.model.add(Convolution1D(nb_filter=7*self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='same', init='he_normal'))
        self.model.add(MaxPooling1D(pool_length=self.poolLength))

        self.model.add(Flatten())
        self.model.add(Dense(200, activation='relu'))
        self.model.add(Dropout(0.25))
        self.model.add(Dense(20, activation='softmax'))

        json_string = self.model.to_json()
        f = open('{0}model.json'.format(self.root), 'w')
        f.write(json_string)
        f.close()
        
        self.model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])

    def defineNetwork2(self):
        print("Setting up network...")
        self.model = Sequential()

        self.model.add(Convolution1D(nb_filter=self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='same', init='he_normal', 
            input_shape=(self.nLambda,1)))        
        self.model.add(MaxPooling1D(pool_length=self.poolLength))

        self.model.add(Convolution1D(nb_filter=2*self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='same', init='he_normal', 
            input_shape=(self.nLambda,1)))        
        self.model.add(MaxPooling1D(pool_length=self.poolLength))

        self.model.add(Convolution1D(nb_filter=3*self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='same', init='he_normal', 
            input_shape=(self.nLambda,1)))        
        self.model.add(MaxPooling1D(pool_length=self.poolLength))

        self.model.add(Convolution1D(nb_filter=4*self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='same', init='he_normal', 
            input_shape=(self.nLambda,1)))        
        self.model.add(MaxPooling1D(pool_length=self.poolLength))
        
        self.model.add(Flatten())
        self.model.add(Dense(500, activation='relu'))
        self.model.add(Dropout(0.25))
        self.model.add(Dense(20, activation='softmax'))

        json_string = self.model.to_json()
        f = open('{0}model.json'.format(self.root), 'w')
        f.write(json_string)
        f.close()
        
        self.model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])


    def trainCNN(self, nIterations):
        print("Training network...")
        self.checkpointer = ModelCheckpoint(filepath="{0}weights.hdf5".format(self.root), verbose=1, save_best_only=True)
        self.history = LossHistory()
        self.metrics = self.model.fit(self.inTrain, self.outTrain, batch_size=self.batchSize, 
            nb_epoch=nIterations, validation_split=0.2, shuffle=False, callbacks=[self.checkpointer, self.history])

        self.history = np.asarray(self.history)
        np.save('{0}losses.npy'.format(self.root), self.history)

        # self.model.fit(self.XTrainSet, self.YTrainSet, batch_size=self.batchSize, nb_epoch=self.nbEpoch, validation_split=0.2)


out = trainDNN('cnns/test3_')
out.readData()
# out.extractTrainingData2()
out.defineNetwork2()
# # out.defineFully()
out.trainCNN(50)
# out.testCNN()