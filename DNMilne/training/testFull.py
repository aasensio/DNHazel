import numpy as np
import platform
import os
from scipy.io import netcdf
import matplotlib.pyplot as pl
from ipdb import set_trace as stop
import seaborn as sns

class doPlotDNN(object):

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

# BField, theta, chi, vmac, damping, B0, B1, doppler, kl
        self.lower = np.asarray([0.0,      0.0,   0.0, -6.0, 0.0,  0.1, 0.1, 0.045,  0.1])
        self.upper = np.asarray([3000.0, 180.0, 180.0,  6.0, 0.5, 20.0, 20.0, 0.100, 20.0])

    def readData(self):
        print("Reading data...")
        self.f = netcdf.netcdf_file('/net/viga/scratch1/deepLearning/DNMilne/database/database_1000000.db', 'r')        
        self.parameters = self.f.variables['parameters'][:,self.left:self.nTrainSamples]
        
        self.outTrain = []        
        for i in range(9):
            self.outTrain.append(np.floor((self.parameters[i,:] - self.lower[i]) / (self.upper[i] - self.lower[i]) * self.nClasses))

        self.prob = np.load("{0}_{1}_prob.npy".format(self.root, self.noise))
        
    def doPlotQuality(self):
        nPars, nPoints, nClasses = self.prob.shape

# BField, theta, chi, vmac, damping, B0, B1, doppler, kl
        labels = ['B [G]', r'$\theta_B$ [deg]', r'$\phi_B$ [deg]', 'v [km/s]', 'a', 'B$_0$', 'B$_1$', r'$\Delta \lambda$ [mA]', r'$\kappa_l$']
        whichPars = [0,1,2,3,4,5,6,7,8]

        
        pl.close('all')
        f, ax = pl.subplots(nrows=3, ncols=3, figsize=(12,10))
        ax = ax.flatten()

        cmap = sns.color_palette()

        for j, i in enumerate(whichPars):
            x = self.parameters[i,-7000:][:,None] * np.ones((1,self.nClasses))
            y = np.linspace(self.lower[i], self.upper[i], self.nClasses)[None,:] * np.ones((7000,1))
            rgba = np.zeros((7000,30,4))
            rgba[:,:,0:3] = cmap[0]
            rgba[:,:,3] = self.prob[i,-7000:,:] * 0.1

            x = x.reshape((7000*30))
            y = y.reshape((7000*30))
            rgba = rgba.reshape((7000*30,4))

            ax[j].scatter(x, y, color=rgba)
            ax[j].set_xlabel("Original {0}".format(labels[j]))
            ax[j].set_ylabel("Recovered {0}".format(labels[j]))
            ax[j].set_xlim([self.lower[i], self.upper[i]])
            ax[j].set_ylim([self.lower[i], self.upper[i]])

        pl.tight_layout()

        pl.savefig("{0}_{1}_testPlot.png".format(self.root, self.noise))
        pl.savefig("{0}_{1}_testPlot.pdf".format(self.root, self.noise))


out = doPlotDNN('cnns/IQUV_v1', 1e-4)
out.readData()
out.doPlotQuality()