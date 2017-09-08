import numpy as np
import matplotlib.pyplot as pl
import h5py
import platform
import os
import pickle
import seaborn as sns
from keras.models import model_from_json
import json
from ipdb import set_trace as stop

class plotDNN(object):

    def __init__(self, root, noise, n_validation):

        self.root = root
        self.noise = noise
        self.batch_size = 256
        
        self.dataFile = "/net/duna/scratch1/aasensio/deepLearning/milne/database/database_6301_hinode_1component.h5"

        f = h5py.File(self.dataFile, 'r')
        self.pars = f.get("parameters")
        self.stokes = f.get("stokes")
        self.n_models, _ = self.pars.shape

        self.std_values = np.load('{0}_normalization.npy'.format(self.root))
        
        self.lower = np.min(self.pars, axis=0)
        self.upper = np.max(self.pars, axis=0)
        
        self.n_validation = n_validation
        self.n_training = int(self.n_models * 0.9)
        self.n_validation = n_validation

    def read_network(self):
        print("Reading previous network...")
                
        f = open('{0}_model.json'.format(self.root), 'r')
        json_string = f.read()
        f.close()

        self.model = model_from_json(json_string)
        self.model.load_weights("{0}_weights.hdf5".format(self.root))

    def forward_network(self):
        inTest = []
        for i in range(4):
            prof = self.stokes[self.n_training:self.n_training+self.n_validation,:,i]
            prof += np.random.normal(loc=0.0, scale=self.noise, size=prof.shape)
            prof /= self.std_values[None,:,i]
            inTest.append(np.atleast_3d(prof).astype('float32'))
        
        self.prob = self.model.predict(inTest, batch_size=self.batch_size, verbose=1)

    def plotLoss(self):
        labels = ['out_BField', 'out_theta', 'out_chi', 'out_vmac', 'out_a', 'out_B0', 'out_B1', 'out_doppler', 'out_kl']
        labelsTxt = ['B', 'theta', 'chi', 'vmac', 'a', 'B0', 'B1', 'doppler', 'kl']

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
        pl.show()


        stop()

    def plot(self):

        pl.close('all')
        f, ax = pl.subplots(nrows=3, ncols=3, figsize=(12,10))
        ax = ax.flatten()

        labelsTxt = ['B [G]', r'$\theta_B$', r'$\phi_B$', r'$v_\mathrm{mac}$', 'a', 'B$_0$', 'B$_1$', r'$\Delta \lambda_D$ [m$\AA$]', r'$\eta$']

        cmap = sns.color_palette()

        for i in range(9):
            prob = self.prob[i][0:self.n_validation,:]
            nCases, nClasses = prob.shape
            x = self.pars[self.n_training:self.n_training+self.n_validation,i][:,None] * np.ones((1,nClasses))
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
        pl.show()
        # pl.savefig("{0}_{1}_comparison.png".format(self.root, self.noise))


if (__name__ == '__main__'):

    root = 'cnns/6301_hinode_1component_v3'
    noise = 1e-4
    out = plotDNN(root, noise, 500)

    out.plotLoss()

    # out.read_network()
    # out.forward_network()

    # out.plot()
    
    # # 
