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
        self.nTrainSamples = 900000
        self.nClasses = 30
        self.noise = noise

        self.validation = 0.1

        self.left = self.nTrainSamples * (1.0-self.validation)

        self.lower = np.asarray([0.05, -5.0, 5.0, 0.0, 0.0, 0.0, -180.0, 0.0, -180.0])
        self.upper = np.asarray([3.0, 5.0, 18.0, 0.5, 1000.0, 180.0, 180.0, 180.0, 180.0])

    def readData(self):
        print("Reading data...")
        self.f = netcdf.netcdf_file('/net/viga/scratch1/deepLearning/DNHazel/database/database_mus_1000000.db', 'r')        
        self.parameters = self.f.variables['parameters'][:,self.left:self.nTrainSamples]

        mu = self.parameters[7,:]
        thB = self.parameters[5,:] * np.pi / 180.0
        phiB = self.parameters[6,:] * np.pi / 180.0

        cosThB = mu * np.cos(thB) + np.sqrt(1.0-mu**2) * np.sin(thB) * np.cos(phiB)
        sinThB = np.sqrt(1.0 - cosThB**2)

        cosPhiB = (mu * np.sin(thB) * np.cos(phiB) - np.sqrt(1.0-mu**2) * np.cos(thB)) / sinThB
        sinPhiB = np.sin(thB) * np.sin(phiB) / sinThB

        self.ThB = np.arctan2(sinThB, cosThB) * 180.0 / np.pi
        self.PhiB = np.arctan2(sinPhiB, cosPhiB) * 180.0 / np.pi
        
        self.outTrain = []        
        for i in range(7):
            self.outTrain.append(self.parameters[i,:])

        self.outTrain.append(self.ThB)
        self.outTrain.append(self.PhiB)

        self.prob = np.load("{0}_{1}_prob.npy".format(self.root, self.noise))
        
    def doPlotQuality(self):
        nPars, nPoints, nClasses = self.prob.shape

        self.mean = np.zeros((nPoints,nPars))
        self.mean2 = np.zeros((nPoints,nPars))
        self.std = np.zeros((nPoints,nPars))

        labels = [r'$\tau$', 'v [km/s]', r'$\Delta v$ [km/s]', 'B [G]', r'$\theta_B$ [deg]', r'$\phi_B$ [deg]', r'$\Theta_B$ [deg]', r'$\Phi_B$ [deg]']
        whichPars = [0,1,2,4,5,6,7,8]

        cmap = sns.color_palette()

        # for i in range(nPars):
        #     xvalues = np.linspace(self.lower[i], self.upper[i], self.nClasses)
        #     if (i == nPars-1):
        #         self.mean[:,i] = np.sum(xvalues[None,0:15] * self.prob[i,:,0:15], axis=1) / np.sum(self.prob[i,:,0:15], axis=1)
        #         self.mean2[:,i] = np.sum(xvalues[None,15:] * self.prob[i,:,15:], axis=1) / np.sum(self.prob[i,:,15:], axis=1)
        #     else:
        #         self.mean[:,i] = np.sum(xvalues[None,:] * self.prob[i,:,:], axis=1)
        #     self.std[:,i] = np.sqrt(np.sum(xvalues**2 * self.prob[i,:,:], axis=1) - self.mean[:,i]**2)
        
        pl.close('all')
        f, ax = pl.subplots(nrows=4, ncols=2, figsize=(8,14))
        ax = ax.flatten()

        

        for j, i in enumerate(whichPars):
            x = self.outTrain[i][-7000:][:,None] * np.ones((1,self.nClasses))
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
        pl.savefig("{0}_{1}_allPars.png".format(self.root, self.noise))

        stop()

        muVals = [1.0, 0.8, 0.6, 0.4, 0.2]
        whichPars = [5,6,7,8]
        labels = [r'$\theta_B$ [deg]', r'$\phi_B$ [deg]', r'$\Theta_B$ [deg]', r'$\Phi_B$ [deg]']

        f, ax = pl.subplots(nrows=5, ncols=4, figsize=(12,12), sharex='col')
        
        for k, muVal in enumerate(muVals):
            mu = self.parameters[-1,-7000:]
            ind = np.where((mu < muVal) & (mu > muVal-0.2))[0]
            n = len(ind)

            for j, i in enumerate(whichPars):
                xx = self.outTrain[i][-7000:] 
                x = xx[ind][:,None] * np.ones((1,self.nClasses))

                y = np.linspace(self.lower[i], self.upper[i], self.nClasses)[None,:] * np.ones((n,1))
                rgba = np.zeros((n,30,4))
                rgba[:,:,0:3] = cmap[0]

                prob = self.prob[i,-7000:,:]
                rgba[:,:,3] = prob[ind,:] * 0.1

                x = x.reshape((n*30))
                y = y.reshape((n*30))
                rgba = rgba.reshape((n*30,4))

                ax[k,j].scatter(x, y, color=rgba)
                if (j == 0):
                    ax[k,j].set_ylabel("Recovered {0}".format(labels[j]))
                ax[k,j].set_xlim([self.lower[i], self.upper[i]])
                ax[k,j].set_ylim([self.lower[i], self.upper[i]])
                ax[k,j].set_title('{0:0.1f} $\leq \mu \leq$ {1:0.1f}'.format(muVal-0.2,muVal))

                ax[-1,j].set_xlabel("Original {0}".format(labels[j]))


        pl.tight_layout()

        pl.savefig("{0}_{1}_angles.png".format(self.root, self.noise))
        # pl.savefig("{0}_{1}_testPlot.pdf".format(self.root, self.noise))


out = doPlotDNN('cnns_mu/IQUV-tau_vth_v_a_B_thB_phiB_LOS_noise1e-4', 1e-4)
out.readData()
out.doPlotQuality()