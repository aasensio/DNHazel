import numpy as np
import matplotlib.pyplot as pl
import h5py
import platform
import os
import pickle
import seaborn as sns
import json
from ipdb import set_trace as stop

class plotDNN(object):

    def __init__(self, root, noise):

        self.root = root
        self.noise = noise
        
        self.dataFile = "/net/duna/scratch1/aasensio/deepLearning/DNStars/database/database.h5"

        f = h5py.File(self.dataFile, 'r')
        self.pars = f.get("parameters")        
        self.nModels, _ = self.pars.shape

        self.lower = np.min(self.pars, axis=0)
        self.upper = np.max(self.pars, axis=0)
        
        self.nTraining = int(self.nModels * 0.9)
        self.nValidation = int(self.nModels * 0.1)

    def readDNNResults(self):
        with open("{0}_{1}_prob.pkl".format(self.root, self.noise), "rb") as outfile:
            self.prob = pickle.load(outfile)

    def plotLoss(self):
        labels = ['out_metals', 'out_O', 'out_vdop', 'out_teff', 'out_logg']
        labelsTxt = ['metals', 'O', 'vdop', 'Teff', 'logg']
        with open("{0}_loss.json".format(self.root), 'r') as f:
            tmp = json.load(f)

        n = len(tmp)

        loss = np.zeros((n,5,2))

        for i in range(5):
            for j in range(n):
                loss[j,i,0] = tmp[j]['{0}_acc'.format(labels[i])]
                loss[j,i,1] = tmp[j]['val_{0}_acc'.format(labels[i])]

        pl.close('all')
        f, ax = pl.subplots(ncols=2, nrows=1, figsize=(12,6))

        for i in range(5):
            ax[0].plot(loss[:,i,0], label=labelsTxt[i])

        ax[0].legend()
        ax[0].set_title('Training set')

        for i in range(5):
            ax[1].plot(loss[:,i,1], label=labelsTxt[i])

        ax[1].legend()
        ax[1].set_title('Validation set')

        pl.tight_layout()


        stop()

    def plot(self):

        pl.close('all')
        f, ax = pl.subplots(nrows=5, ncols=1, figsize=(5,16))
        ax = ax.flatten()

        labelsTxt = ['[Fe/H]', r'[$\alpha$/Fe]', r'log(v$_\mathrm{mic}$)', r'T$_\mathrm{eff}$', 'log g']

        cmap = sns.color_palette()

        for i in range(5):
            prob = self.prob[i][0:1000,:]
            nCases, nClasses = prob.shape
            x = self.pars[0:1000,i][:,None] * np.ones((1,nClasses))
            # x = self.pars[0:1000,i][:,None] * np.ones((1,nClasses))
            y = np.linspace(self.lower[i], self.upper[i], nClasses)[None,:] * np.ones((nCases,1))
            rgba = np.zeros((nCases,nClasses,4))
            rgba[:,:,0:3] = cmap[0]
            rgba[:,:,3] = prob * 0.1

            x = x.reshape((nCases*nClasses))
            y = y.reshape((nCases*nClasses))
            rgba = rgba.reshape((nCases*nClasses,4))

            ax[i].scatter(x, y, color=rgba)
            ax[i].set_xlabel("Original {0}".format(labelsTxt[i]))
            ax[i].set_ylabel("Recovered {0}".format(labelsTxt[i]))
            ax[i].set_xlim([self.lower[i], self.upper[i]])
            ax[i].set_ylim([self.lower[i], self.upper[i]])

        pl.tight_layout()


if (__name__ == '__main__'):

    root = 'cnns/test'
    noise = '1e-4'
    out = plotDNN(root, noise)

    # out.plotLoss()

    out.readDNNResults()

    out.plot()

    # 
