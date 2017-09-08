import numpy as np
import platform
import os
import simplejson
import sys

if (platform.node() == 'viga'):
    os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32"

os.environ["KERAS_BACKEND"] = "theano"

from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Convolution1D, MaxPooling1D, Flatten, merge, GaussianNoise
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
        self.nLambda = 128
        self.batchSize = 128
        self.nImages = 10
        self.nTrainSamples = 900000
        self.nClasses = 30

        self.lower = np.asarray([0.05, -5.0, 5.0, 0.0, 0.0, 0.0, -180.0, 0.0, -180.0])
        self.upper = np.asarray([3.0, 5.0, 18.0, 0.5, 1000.0, 180.0, 180.0, 180.0, 180.0])

    def readData(self):
        print("Reading data...")
        self.f = netcdf.netcdf_file('/net/viga/scratch1/deepLearning/DNHazel/database/database_mus_1000000.db', 'r')
        self.stokes = self.f.variables['stokes'][:]
        self.parameters = self.f.variables['parameters'][:]

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
        for i in range(4):            
            self.inTrain.append(self.stokes[i,:,0:self.nTrainSamples].T.reshape((self.nTrainSamples, self.nLambda, 1)).astype('float32'))

        self.inTrain.append(self.parameters[-1,0:self.nTrainSamples].reshape((self.nTrainSamples, 1)).astype('float32'))
        
        self.outTrain = []
        for i in range(7):
            outTrain = np.floor((self.parameters[i,0:self.nTrainSamples] - self.lower[i]) / (self.upper[i] - self.lower[i]) * self.nClasses).astype('int32')
            self.outTrain.append(np_utils.to_categorical(outTrain, self.nClasses))

# Add outputs for LOS angles
        outTrain = np.floor((ThB[0:self.nTrainSamples] - self.lower[7]) / (self.upper[7] - self.lower[7]) * self.nClasses).astype('int32')
        self.outTrain.append(np_utils.to_categorical(outTrain, self.nClasses))

        outTrain = np.floor((PhiB[0:self.nTrainSamples] - 0.001 - self.lower[8]) / (self.upper[8] - self.lower[8]) * self.nClasses).astype('int32')
        self.outTrain.append(np_utils.to_categorical(outTrain, self.nClasses))

        self.f.close()
        
    def defineNetwork(self):
        print("Setting up network...")

# Stokes I
        sI_input = Input(shape=(self.nLambda,1), name='SI_input')
        x = GaussianNoise(sigma=1e-4)(sI_input)

        x = Convolution1D(nb_filter=self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='convI_1')(x)
        x = Convolution1D(nb_filter=self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='convI_1_2', subsample_length=2)(x)

        x = Convolution1D(nb_filter=self.nFeatures, filter_length=2*self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='convI_2')(x)
        x = Convolution1D(nb_filter=self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='convI_2_2', subsample_length=2)(x)

        x = Convolution1D(nb_filter=self.nFeatures, filter_length=3*self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='convI_3')(x)
        x = Convolution1D(nb_filter=self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='convI_3_2', subsample_length=2)(x)

        x = Convolution1D(nb_filter=self.nFeatures, filter_length=4*self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='convI_4')(x)
        x = Convolution1D(nb_filter=self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='convI_4_2', subsample_length=2)(x)

        x = Convolution1D(nb_filter=self.nFeatures, filter_length=5*self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='convI_5')(x)
        x = Convolution1D(nb_filter=self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='convI_5_2', subsample_length=2)(x)
        stI_intermediate = Flatten(name='flatI')(x)

# Stokes Q
        sQ_input = Input(shape=(self.nLambda,1), name='SQ_input')
        x = GaussianNoise(sigma=1e-4)(sQ_input)

        x = Convolution1D(nb_filter=self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='convQ_1')(x)
        x = Convolution1D(nb_filter=self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='convQ_1_2', subsample_length=2)(x)

        x = Convolution1D(nb_filter=self.nFeatures, filter_length=2*self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='convQ_2')(x)
        x = Convolution1D(nb_filter=self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='convQ_2_2', subsample_length=2)(x)

        x = Convolution1D(nb_filter=self.nFeatures, filter_length=3*self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='convQ_3')(x)
        x = Convolution1D(nb_filter=self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='convQ_3_2', subsample_length=2)(x)

        x = Convolution1D(nb_filter=self.nFeatures, filter_length=4*self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='convQ_4')(x)
        x = Convolution1D(nb_filter=self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='convQ_4_2', subsample_length=2)(x)

        x = Convolution1D(nb_filter=self.nFeatures, filter_length=5*self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='convQ_5')(x)
        x = Convolution1D(nb_filter=self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='convQ_5_2', subsample_length=2)(x)
        stQ_intermediate = Flatten(name='flatQ')(x)

# Stokes U
        sU_input = Input(shape=(self.nLambda,1), name='SU_input')
        x = GaussianNoise(sigma=1e-4)(sU_input)

        x = Convolution1D(nb_filter=self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='convU_1')(x)
        x = Convolution1D(nb_filter=self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='convU_1_2', subsample_length=2)(x)

        x = Convolution1D(nb_filter=self.nFeatures, filter_length=2*self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='convU_2')(x)
        x = Convolution1D(nb_filter=self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='convU_2_2', subsample_length=2)(x)

        x = Convolution1D(nb_filter=self.nFeatures, filter_length=3*self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='convU_3')(x)
        x = Convolution1D(nb_filter=self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='convU_3_2', subsample_length=2)(x)

        x = Convolution1D(nb_filter=self.nFeatures, filter_length=4*self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='convU_4')(x)
        x = Convolution1D(nb_filter=self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='convU_4_2', subsample_length=2)(x)

        x = Convolution1D(nb_filter=self.nFeatures, filter_length=5*self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='convU_5')(x)
        x = Convolution1D(nb_filter=self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='convU_5_2', subsample_length=2)(x)
        stU_intermediate = Flatten(name='flatU')(x)

# Stokes V
        sV_input = Input(shape=(self.nLambda,1), name='SV_input')
        x = GaussianNoise(sigma=1e-4)(sV_input)

        x = Convolution1D(nb_filter=self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='convV_1')(x)
        x = Convolution1D(nb_filter=self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='convV_1_2', subsample_length=2)(x)

        x = Convolution1D(nb_filter=self.nFeatures, filter_length=2*self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='convV_2')(x)
        x = Convolution1D(nb_filter=self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='convV_2_2', subsample_length=2)(x)

        x = Convolution1D(nb_filter=self.nFeatures, filter_length=3*self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='convV_3')(x)
        x = Convolution1D(nb_filter=self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='convV_3_2', subsample_length=2)(x)

        x = Convolution1D(nb_filter=self.nFeatures, filter_length=4*self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='convV_4')(x)
        x = Convolution1D(nb_filter=self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='convV_4_2', subsample_length=2)(x)

        x = Convolution1D(nb_filter=self.nFeatures, filter_length=5*self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='convV_5')(x)
        x = Convolution1D(nb_filter=self.nFeatures, filter_length=self.kernelSize, activation='relu', border_mode='same', init='he_normal', name='convV_5_2', subsample_length=2)(x)
        stV_intermediate = Flatten(name='flatV')(x)

        mu_input = Input(shape=(1,), name='mu_input')

        intermediate_conv = merge([stI_intermediate, stQ_intermediate, stU_intermediate, stV_intermediate, mu_input], mode='concat', name='FC')

        x = Dense(5*self.nClasses, activation='relu', name='FC_tau')(intermediate_conv)
        out_tau = Dense(self.nClasses, activation='softmax', name='out_tau')(x)
        out_v = Dense(self.nClasses, activation='softmax', name='out_v')(x)
        out_vth = Dense(self.nClasses, activation='softmax', name='out_vth')(x)
        out_a = Dense(self.nClasses, activation='softmax', name='out_a')(x)

        x = Dense(5*self.nClasses, activation='relu', name='FC_B')(intermediate_conv)
        out_B = Dense(self.nClasses, activation='softmax', name='out_B')(x)
        out_thB = Dense(self.nClasses, activation='softmax', name='out_thB')(x)        
        out_phiB = Dense(self.nClasses, activation='softmax', name='out_phiB')(x)
        out_ThB = Dense(self.nClasses, activation='softmax', name='out_ThB')(x)        
        out_PhiB = Dense(self.nClasses, activation='softmax', name='out_PhiB')(x)

        
        self.model = Model(input=[sI_input, sQ_input, sU_input, sV_input, mu_input], output=[out_tau, out_v, out_vth, out_a, out_B, out_thB, out_phiB, out_ThB, out_PhiB])

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
        
        # self.model.fit(self.XTrainSet, self.YTrainSet, batch_size=self.batchSize, nb_epoch=self.nbEpoch, validation_split=0.2)


if (__name__ == '__main__'):
    if (len(sys.argv) != 4):
        print("Usage:")
        print("python trainStI.py root_file nEpochs start/continue")
        print("Example:")
        print("python trainIQUV_mu.py cnns_mu/IQUV-tau_vth_v_a_B_thB_phiB 20 start")
        sys.exit()
    else:
        root = sys.argv[1]
        nEpochs = int(sys.argv[2])
        option = sys.argv[3]
        out = trainDNNFull(root)
        if (option == 'start'):            
            out.defineNetwork()
        elif (option == 'continue'):
            out.readNetwork()
        else:
            print("Option {0} not correct".format(option))
            sys.exit()
        out.readData()
        out.trainCNN(nEpochs)
