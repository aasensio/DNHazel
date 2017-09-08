import numpy as np
import platform
import os
if (platform.node() == 'viga'):
    os.environ["THEANO_FLAGS"] = "device=cpu,floatX=float32"
from keras.models import Sequential
from keras.layers import Dense, Dropout, Convolution1D, MaxPooling1D, Flatten
from keras.callbacks import ModelCheckpoint, Callback
from keras.utils import np_utils
from keras.models import model_from_json
from scipy.io import netcdf
import matplotlib.pyplot as pl
import pyhazel
from ipdb import set_trace as stop

def i0Allen(wavelength, muAngle):
    """
    Return the solar intensity at a specific wavelength and heliocentric angle
    wavelength: wavelength in angstrom
    muAngle: cosine of the heliocentric angle
    """
    C = 2.99792458e10
    H = 6.62606876e-27

    lambdaIC = 1e4 * np.asarray([0.20,0.22,0.245,0.265,0.28,0.30,0.32,0.35,0.37,0.38,0.40,0.45,0.50,0.55,0.60,0.80,1.0,1.5,2.0,3.0,5.0,10.0])
    uData = np.asarray([0.12,-1.3,-0.1,-0.1,0.38,0.74,0.88,0.98,1.03,0.92,0.91,0.99,0.97,0.93,0.88,0.73,0.64,0.57,0.48,0.35,0.22,0.15])
    vData = np.asarray([0.33,1.6,0.85,0.90,0.57, 0.20, 0.03,-0.1,-0.16,-0.05,-0.05,-0.17,-0.22,-0.23,-0.23,-0.22,-0.20,-0.21,-0.18,-0.12,-0.07,-0.07])

    lambdaI0 = 1e4 * np.asarray([0.20,0.22,0.24,0.26,0.28,0.30,0.32,0.34,0.36,0.37,0.38,0.39,0.40,0.41,0.42,0.43,0.44,0.45,0.46,0.48,0.50,0.55,0.60,0.65,0.70,0.75,\
        0.80,0.90,1.00,1.10,1.20,1.40,1.60,1.80,2.00,2.50,3.00,4.00,5.00,6.00,8.00,10.0,12.0])
    I0 = np.asarray([0.06,0.21,0.29,0.60,1.30,2.45,3.25,3.77,4.13,4.23,4.63,4.95,5.15,5.26,5.28,5.24,5.19,5.10,5.00,4.79,4.55,4.02,3.52,3.06,2.69,2.28,2.03,\
        1.57,1.26,1.01,0.81,0.53,0.36,0.238,0.160,0.078,0.041,0.0142,0.0062,0.0032,0.00095,0.00035,0.00018])
    I0 *= 1e14 * (lambdaI0 * 1e-8)**2 / C

    u = np.interp(wavelength, lambdaIC, uData)
    v = np.interp(wavelength, lambdaIC, vData)
    i0 = np.interp(wavelength, lambdaI0, I0)
    
    return (1.0 - u - v + u * muAngle + v * muAngle**2)* i0

class trainDNN(object):

    def __init__(self, root, noise):
        self.root = root
        self.nFeatures = 50
        self.kernelSize = 5
        self.poolLength = 2
        self.nLambda = 128
        self.batchSize = 128
        self.nImages = 10
        self.nTrainSamples = 900000
        self.nClasses = 30
        self.noise = noise
        self.validation = 0.1

        self.left = int(self.nTrainSamples * (1.0-self.validation))

        self.lower = np.asarray([0.05, -5.0, 5.0, 0.0, 0.0, 0.0, -180.0, 0.0])
        self.upper = np.asarray([3.0, 5.0, 18.0, 0.5, 1000.0, 180.0, 180.0, 1.0])

        pyhazel.init()

    def compute(self, pars):

        nLambdaInput = 128
        GRIS_dispersion = 0.0362  # A/pix
        lowerLambda = 10828
        upperLambda = lowerLambda + GRIS_dispersion * nLambdaInput

        tau, v, vth, a, B, theta, phi, mu = pars

        synModeInput = 5
        nSlabsInput = 1

        B1Input = np.asarray([B, theta, phi])    
        B2Input = np.asarray([0.0,0.0,0.0])
        
        hInput = 3.e0

        tau1Input = tau
        tau2Input = 0.e0

        I0 = i0Allen(10830.0, mu)

        boundaryInput  = np.asarray([I0,0.0,0.0,0.0])

        transInput = 1
        atomicPolInput = 1

        anglesInput = np.asarray([np.arccos(mu)*180/np.pi,0.0,0.0])

        lambdaAxisInput = np.linspace(lowerLambda-10829.0911, upperLambda-10829.0911, nLambdaInput)        

        dopplerWidthInput = vth
        dopplerWidth2Input = 0.e0

        dampingInput = a

        dopplerVelocityInput = v
        dopplerVelocity2Input = 0.e0

        ffInput = 0.e0
        betaInput = 1.0
        beta2Input = 1.0
        nbarInput = np.asarray([0.0,0.0,0.0,0.0])
        omegaInput = np.asarray([0.0,0.0,0.0,0.0])
        
        nbarInput = np.asarray([1.0,1.0,1.0,1.0])
        omegaInput = np.asarray([1.0,1.0,1.0,1.0])
        normalization = 0
        
        # Compute the Stokes parameters using many default parameters, using Allen's data
        [l, stokes, etaOutput, epsOutput] = pyhazel.synth(synModeInput, nSlabsInput, B1Input, B2Input, hInput, 
                                tau1Input, tau2Input, boundaryInput, transInput, atomicPolInput, anglesInput, 
                                nLambdaInput, lambdaAxisInput, dopplerWidthInput, dopplerWidth2Input, dampingInput, 
                                dopplerVelocityInput, dopplerVelocity2Input, ffInput, betaInput, beta2Input, nbarInput, omegaInput, normalization)

        return l, stokes

    def readData(self):
        print("Reading data...")
        self.f = netcdf.netcdf_file('/net/viga/scratch1/deepLearning/DNHazel/database/database_mus_1000000.db', 'r')
        self.stokes = self.f.variables['stokes'][:].copy()
        self.parameters = self.f.variables['parameters'][:].copy()
        self.f.close()

        self.inTrain = []
        for i in range(4):
            stokes = self.stokes[i,:,self.left:self.nTrainSamples].T.reshape((int(self.validation*self.nTrainSamples), self.nLambda, 1))
            stokes += self.noise * np.random.randn(int(self.validation*self.nTrainSamples), self.nLambda, 1)
            self.inTrain.append(stokes.astype('float32'))
            
        self.inTrain.append(self.parameters[-1,self.left:self.nTrainSamples].reshape((int(self.validation*self.nTrainSamples), 1)).astype('float32'))

        self.prob = np.load("{0}_{1}_prob.npy".format(self.root, self.noise))
    
    def defineNetwork(self):
        print("Setting up network...")
        
        f = open('{0}_model.json'.format(self.root), 'r')
        json_string = f.read()
        f.close()

        self.model = model_from_json(json_string)
        self.model.load_weights("{0}_weights.hdf5".format(self.root))

    def testCNN(self):
        print("Computing probabilities...")
        
        pars = self.parameters[:,self.left]
        mu = pars[-1]
        lam, sto = self.compute(pars.astype('float64'))

        inTrain = []
        for i in range(4):
            stokes = np.atleast_3d(sto[i,:]).reshape((1, self.nLambda, 1))
            stokes += self.noise * np.random.randn(1,self.nLambda,1)
            inTrain.append(stokes.astype('float32'))
        inTrain.append(np.atleast_2d(mu).reshape((1, 1)).astype('float32'))

        self.inference = self.model.predict(inTrain, verbose=1, batch_size=1)

        print(self.inference[0])
        print(self.prob[0,0,:])

        stop()

        
        return 

out = trainDNN('cnns_mu/IQUV-tau_vth_v_a_B_thB_phiB', 1e-4)
out.readData()
out.defineNetwork()
prob = out.testCNN()
