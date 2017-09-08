import numpy as np
import platform
import os
if (platform.node() == 'viga'):
    os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32"
from keras.models import Sequential
from keras.layers import Dense, Dropout, Convolution1D, MaxPooling1D, Flatten
from keras.callbacks import ModelCheckpoint, Callback
from keras.utils import np_utils
from keras.models import model_from_json
from scipy.io import netcdf
import matplotlib.pyplot as pl
# from ipdb import set_trace as stop

class trainDNN(object):

    def __init__(self, root, noise):
        self.root = root
        self.nFeatures = 50
        self.kernelSize = 5
        self.poolLength = 2
        self.nLambda = 64
        self.batchSize = 128
        self.nImages = 10
        self.nTrainSamples = 900000
        self.nClasses = 30
        self.noise = noise
        self.validation = 0.1

        self.left = self.nTrainSamples * (1.0-self.validation)

    def readData(self):
        print("Reading data...")
        self.f = netcdf.netcdf_file('/net/viga/scratch1/deepLearning/DNMilne/database/database_1000000.db', 'r')
        self.stokes = self.f.variables['stokes'][:].copy()
        self.f.close()

        self.inTrain = []
        for i in range(4):
            stokes = self.stokes[i,:,self.left:self.nTrainSamples].T.reshape((self.validation*self.nTrainSamples, self.nLambda, 1))
            stokes += self.noise * np.random.randn(self.validation*self.nTrainSamples, self.nLambda, 1)
            self.inTrain.append(stokes)
                
    def defineNetwork(self):
        print("Setting up network...")
        
        f = open('{0}_model.json'.format(self.root), 'r')
        json_string = f.read()
        f.close()

        self.model = model_from_json(json_string)
        self.model.load_weights("{0}_weights.hdf5".format(self.root))

    def testCNN(self):
        print("Computing probabilities...")
        out = self.model.predict(self.inTrain, verbose=1)

        np.save("{0}_{1}_prob.npy".format(self.root, self.noise), out)
        return 

out = trainDNN('cnns/IQUV_v1', 1e-3)
out.readData()
out.defineNetwork()
prob = out.testCNN()
