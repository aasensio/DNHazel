import numpy as np
import platform
import os
import simplejson
import sys

if (platform.node() == 'viga'):
    os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32"
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Convolution1D, MaxPooling1D, Flatten, merge
from keras.callbacks import ModelCheckpoint, Callback
from keras.models import Model, model_from_json
from keras.utils.visualize_util import plot as kerasPlot
import keras.optimizers
# from keras.optimizers import Nadam
from keras.utils import np_utils
from scipy.io import netcdf

class LossHistory(Callback):
    def __init__(self, root):
        self.root = root
        self.fHistory = open("{0}_loss.json".format(self.root), 'w')

    def on_train_begin(self, logs={}):        
        self.losses = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs)
        simplejson.dump(logs, self.fHistory, indent=2)

    def finalize(self):
        self.fHistory.close()

class trainDNNFull(object):

    def __init__(self, root):
        self.root = root
        self.nFeatures = 50
        self.kernelSize = 5
        self.poolLength = 2
        self.nLambda = 64
        self.batchSize = 128
        self.nImages = 10
        self.nTrainSamples = 900000
        self.nClasses = 30

# BField, theta, chi, vmac, damping, B0, B1, doppler, kl
        self.lower = np.asarray([0.0,      0.0,   0.0, -6.0, 0.0,  0.1, 0.1, 0.045,  0.1])
        self.upper = np.asarray([3000.0, 180.0, 180.0,  6.0, 0.5, 20.0, 20.0, 0.100, 20.0])
        
    def readData(self):
        print("Reading data...")
        self.f = netcdf.netcdf_file('/net/viga/scratch1/deepLearning/DNMilne/database/database_1000000.db', 'r')
        self.stokes = self.f.variables['stokes'][:]
        self.parameters = self.f.variables['parameters'][:]

        self.inTrain = []
        for i in range(4):            
            self.inTrain.append(self.stokes[i,:,0:self.nTrainSamples].T.reshape((self.nTrainSamples, self.nLambda, 1)))
        
        self.outTrain = []
        for i in range(9):
            outTrain = np.floor((self.parameters[i,0:self.nTrainSamples] - self.lower[i]) / (self.upper[i] - self.lower[i]) * self.nClasses).astype('int32')
            self.outTrain.append(np_utils.to_categorical(outTrain, self.nClasses))

        self.f.close()
        
    def defineNetwork(self):
        print("Setting up network...")

# Stokes I
        sI_input = Input(shape=(self.nLambda,1), name='SI_input')
        x = Convolution1D(nb_filter=self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='convI_1')(sI_input)
        x = MaxPooling1D(pool_length=self.poolLength, name='poolI_1')(x)

        x = Convolution1D(nb_filter=self.nFeatures, filter_length=2*self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='convI_2')(x)
        x = MaxPooling1D(pool_length=self.poolLength, name='poolI_2')(x)

        x = Convolution1D(nb_filter=self.nFeatures, filter_length=3*self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='convI_3')(x)
        x = MaxPooling1D(pool_length=self.poolLength, name='poolI_3')(x)

        x = Convolution1D(nb_filter=self.nFeatures, filter_length=4*self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='convI_4')(x)
        x = MaxPooling1D(pool_length=self.poolLength, name='poolI_4')(x)

        x = Convolution1D(nb_filter=self.nFeatures, filter_length=5*self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='convI_5')(x)
        x = MaxPooling1D(pool_length=self.poolLength, name='poolI_5')(x)
        stI_intermediate = Flatten(name='flatI')(x)

# Stokes Q
        sQ_input = Input(shape=(self.nLambda,1), name='SQ_input')
        x = Convolution1D(nb_filter=self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='convQ_1')(sQ_input)
        x = MaxPooling1D(pool_length=self.poolLength, name='poolQ_1')(x)

        x = Convolution1D(nb_filter=self.nFeatures, filter_length=2*self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='convQ_2')(x)
        x = MaxPooling1D(pool_length=self.poolLength, name='poolQ_2')(x)

        x = Convolution1D(nb_filter=self.nFeatures, filter_length=3*self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='convQ_3')(x)
        x = MaxPooling1D(pool_length=self.poolLength, name='poolQ_3')(x)

        x = Convolution1D(nb_filter=self.nFeatures, filter_length=4*self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='convQ_4')(x)
        x = MaxPooling1D(pool_length=self.poolLength, name='poolQ_4')(x)

        x = Convolution1D(nb_filter=self.nFeatures, filter_length=5*self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='convQ_5')(x)
        x = MaxPooling1D(pool_length=self.poolLength, name='poolQ_5')(x)
        stQ_intermediate = Flatten(name='flatQ')(x)

# Stokes U
        sU_input = Input(shape=(self.nLambda,1), name='SU_input')
        x = Convolution1D(nb_filter=self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='convU_1')(sU_input)
        x = MaxPooling1D(pool_length=self.poolLength, name='poolU_1')(x)

        x = Convolution1D(nb_filter=self.nFeatures, filter_length=2*self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='convU_2')(x)
        x = MaxPooling1D(pool_length=self.poolLength, name='poolU_2')(x)

        x = Convolution1D(nb_filter=self.nFeatures, filter_length=3*self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='convU_3')(x)
        x = MaxPooling1D(pool_length=self.poolLength, name='poolU_3')(x)

        x = Convolution1D(nb_filter=self.nFeatures, filter_length=4*self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='convU_4')(x)
        x = MaxPooling1D(pool_length=self.poolLength, name='poolU_4')(x)

        x = Convolution1D(nb_filter=self.nFeatures, filter_length=5*self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='convU_5')(x)
        x = MaxPooling1D(pool_length=self.poolLength, name='poolU_5')(x)
        stU_intermediate = Flatten(name='flatU')(x)

# Stokes V
        sV_input = Input(shape=(self.nLambda,1), name='SV_input')
        x = Convolution1D(nb_filter=self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='convV_1')(sV_input)
        x = MaxPooling1D(pool_length=self.poolLength, name='poolV_1')(x)

        x = Convolution1D(nb_filter=self.nFeatures, filter_length=2*self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='convV_2')(x)
        x = MaxPooling1D(pool_length=self.poolLength, name='poolV_2')(x)

        x = Convolution1D(nb_filter=self.nFeatures, filter_length=3*self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='convV_3')(x)
        x = MaxPooling1D(pool_length=self.poolLength, name='poolV_3')(x)

        x = Convolution1D(nb_filter=self.nFeatures, filter_length=4*self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='convV_4')(x)
        x = MaxPooling1D(pool_length=self.poolLength, name='poolV_4')(x)

        x = Convolution1D(nb_filter=self.nFeatures, filter_length=5*self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='convV_5')(x)
        x = MaxPooling1D(pool_length=self.poolLength, name='poolV_5')(x)
        stV_intermediate = Flatten(name='flatV')(x)

        intermediate_conv = merge([stI_intermediate, stQ_intermediate, stU_intermediate, stV_intermediate], mode='concat', name='FC')


        x = Dense(10*self.nClasses, activation='relu', name='FC2')(intermediate_conv)
        out_BField = Dense(self.nClasses, activation='softmax', name='out_BField')(x)
        out_theta = Dense(self.nClasses, activation='softmax', name='out_theta')(x)
        out_chi = Dense(self.nClasses, activation='softmax', name='out_chi')(x)
        out_vmac = Dense(self.nClasses, activation='softmax', name='out_vmac')(x)
        out_a = Dense(self.nClasses, activation='softmax', name='out_a')(x)
        out_B0 = Dense(self.nClasses, activation='softmax', name='out_B0')(x)
        out_B1 = Dense(self.nClasses, activation='softmax', name='out_B1')(x)
        out_doppler = Dense(self.nClasses, activation='softmax', name='out_doppler')(x)
        out_kl = Dense(self.nClasses, activation='softmax', name='out_kl')(x)

        self.model = Model(input=[sI_input, sQ_input, sU_input, sV_input], output=[out_BField, out_theta, out_chi, out_vmac,
            out_a, out_B0, out_B1, out_doppler, out_kl])

        json_string = self.model.to_json()
        f = open('{0}_model.json'.format(self.root), 'w')
        f.write(json_string)
        f.close()

        kerasPlot(self.model, to_file='{0}_model.png'.format(self.root), show_shapes=True)
            
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
            nb_epoch=nIterations, validation_split=0.1, shuffle=False, callbacks=[self.checkpointer, self.history])
        self.history.finalize()
        

if (__name__ == '__main__'):
    if (len(sys.argv) != 4):
        print("Usage:")
        print("python trainIQUV.py root_file nEpochs start/continue")
        print("Example:")
        print("python trainIQUV.py cnns/IQUV_v1 20 start")
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
