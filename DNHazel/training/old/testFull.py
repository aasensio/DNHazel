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
        self.nLambda = 128
        self.batchSize = 128
        self.nImages = 10
        self.nTrainSamples = 90000
        self.nClasses = 30
        self.noise = noise

        self.lower = np.asarray([0.05, -5.0, 5.0, 0.0, 0.0, 0.0, -180.0])
        self.upper = np.asarray([3.0, 5.0, 18.0, 0.5, 1000.0, 180.0, 180.0])

    def readData(self):
        print("Reading data...")
        self.f = netcdf.netcdf_file('/net/viga/scratch1/deepLearning/DNHazel/database/database_mu1_100000.db', 'r')
        self.stokes = self.f.variables['stokes'][:,:,0:self.nTrainSamples]
        self.parameters = self.f.variables['parameters'][:,0:self.nTrainSamples]
        
        self.outTrain = []        
        for i in range(7):
            self.outTrain.append(np.floor((self.parameters[i,:] - self.lower[i]) / (self.upper[i] - self.lower[i]) * self.nClasses))

        self.prob = np.load("{0}_{1}_prob.npy".format(self.root, self.noise))
        
    def doPlotQuality(self):
        nPars, nPoints, nClasses = self.prob.shape

        self.mean = np.zeros((nPoints,nPars))
        self.mean2 = np.zeros((nPoints,nPars))
        self.std = np.zeros((nPoints,nPars))

        labels = ['$\tau$', 'v [km/s]', r'$\Delta v$ [km/s]', 'B [G]', r'$\theta_B$ [deg]', r'$\phi_B$ [deg]']
        whichPars = [0,1,2,4,5,6]

        for i in range(nPars):
            xvalues = np.linspace(self.lower[i], self.upper[i], self.nClasses)
            if (i == nPars-1):
                self.mean[:,i] = np.sum(xvalues[None,0:15] * self.prob[i,:,0:15], axis=1) / np.sum(self.prob[i,:,0:15], axis=1)
                self.mean2[:,i] = np.sum(xvalues[None,15:] * self.prob[i,:,15:], axis=1) / np.sum(self.prob[i,:,15:], axis=1)
            else:
                self.mean[:,i] = np.sum(xvalues[None,:] * self.prob[i,:,:], axis=1)
            self.std[:,i] = np.sqrt(np.sum(xvalues**2 * self.prob[i,:,:], axis=1) - self.mean[:,i]**2)
        
        pl.close('all')
        f, ax = pl.subplots(nrows=2, ncols=3, figsize=(12,8))
        ax = ax.flatten()

        for i in range(6):
            ax[i].plot(self.parameters[whichPars[i],-7000:], self.mean[-7000:,whichPars[i]], '.', alpha=0.05)
            ax[i].set_xlabel("Original {0}".format(labels[i]))
            ax[i].set_ylabel("Recovered {0}".format(labels[i]))

        ax[-1].plot(self.parameters[whichPars[i],-7000:], self.mean2[-7000:,whichPars[i]], '.', alpha=0.05)

        pl.tight_layout()
        pl.savefig("{0}_{1}_testPlot.png".format(self.root, self.noise))
        pl.savefig("{0}_{1}_testPlot.pdf".format(self.root, self.noise))


out = doPlotDNN('cnns/IQUV-tau_vth_v_a_B_thB_phiB_v3', 1e-3)
out.readData()
out.doPlotQuality()