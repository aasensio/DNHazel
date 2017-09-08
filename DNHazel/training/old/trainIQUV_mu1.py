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
from ipdb import set_trace as stop

class LossHistory(Callback):
    def __init__(self, root):
        self.root = root
        self.fHistory = open("{0}_loss.json".format(self.root), 'w')

    def on_train_begin(self, logs={}):        
        self.losses = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append([logs.get('loss'), logs.get('val_loss'), logs.get('acc'), logs.get('val_acc')])
        simplejson.dump([logs.get('loss'), logs.get('val_loss'), logs.get('acc'), logs.get('val_acc')], self.fHistory, indent=2)

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
        self.nClasses = 30

        self.lower = np.asarray([0.05, -5.0, 5.0, 0.0, 0.0, 0.0, -180.0])
        self.upper = np.asarray([3.0, 5.0, 18.0, 0.5, 1000.0, 180.0, 180.0])        

    def readData(self):
        print("Reading data...")
        self.f = netcdf.netcdf_file('/net/viga/scratch1/deepLearning/DNHazel/database/database_mu1_100000.db', 'r')
        self.stokes = self.f.variables['stokes'][:]
        self.parameters = self.f.variables['parameters'][:]

        self.inTrain = []
        for i in range(4):            
            self.inTrain.append(self.stokes[i,:,0:self.nTrainSamples].T.reshape((self.nTrainSamples, self.nLambda, 1)))
        
        self.outTrain = []
        for i in range(7):
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

        x = Dense(3*self.nClasses, activation='relu', name='FC_tau')(intermediate_conv)
        out_tau = Dense(self.nClasses, activation='softmax', name='out_tau')(x)
        out_v = Dense(self.nClasses, activation='softmax', name='out_v')(x)
        out_vth = Dense(self.nClasses, activation='softmax', name='out_vth')(x)
        out_a = Dense(self.nClasses, activation='softmax', name='out_a')(x)

        x = Dense(3*self.nClasses, activation='relu', name='FC_B')(intermediate_conv)
        out_B = Dense(self.nClasses, activation='softmax', name='out_B')(x)
        out_thB = Dense(self.nClasses, activation='softmax', name='out_thB')(x)        
        out_phiB = Dense(self.nClasses, activation='softmax', name='out_phiB')(x)

        self.model = Model(input=[sI_input, sQ_input, sU_input, sV_input], output=[out_tau, out_v, out_vth, out_a, out_B, out_thB, out_phiB])

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
