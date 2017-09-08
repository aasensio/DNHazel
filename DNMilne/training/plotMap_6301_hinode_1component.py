import numpy as np
import matplotlib.pyplot as pl
import h5py
import platform
import os
import pickle
import scipy.io as io
import seaborn as sns
from keras.models import model_from_json
import json
from ipdb import set_trace as stop

class plot_map(object):

    def __init__(self, root):

        self.root = root
        self.noise = noise
        self.batch_size = 256

        self.dataFile = "/net/duna/scratch1/aasensio/deepLearning/milne/database/database_6301_hinode_1component.h5"

        f = h5py.File(self.dataFile, 'r')
        self.pars = f.get("parameters")                
        self.lower = np.min(self.pars, axis=0)
        self.upper = np.max(self.pars, axis=0)
        f.close()

        
        self.root_hinode = "/net/nas4/fis/aasensio/scratch/HINODE/SUNSPOT/"

        self.label_files = ["sunspot_stokesI_512x512.sav", "sunspot_stokesQ_512x512.sav", "sunspot_stokesU_512x512.sav", "sunspot_stokesV_512x512.sav"]

        self.std_values = np.load('{0}_normalization.npy'.format(self.root))

        labels_data = ['data_ii', 'data_qq', 'data_uu', 'data_vv']

        self.stokes = np.zeros((512,512,50,4))

        for i in range(4):
            print("Reading file {0}".format(self.label_files[i]))
            stokes = io.readsav("/net/nas4/fis/aasensio/scratch/HINODE/SUNSPOT/{0}".format(self.label_files[i]))[labels_data[i]]
            if (i == 0):
                mean_stokesi = np.mean(stokes[400:500,0:100,0])

            stokes = stokes[:,:,0:50] / mean_stokesi
            self.stokes[:,:,:,i] = stokes / self.std_values[None,None,:,i]

        self.stokes = self.stokes.reshape((512*512,50,4))

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
            inTest.append(np.atleast_3d(self.stokes[:,:,i]).astype('float32'))
        
        self.prob = self.model.predict(inTest, batch_size=self.batch_size, verbose=1)

    def plot(self):

        pl.close('all')
        f, ax = pl.subplots(nrows=3, ncols=3, figsize=(12,10))
        ax = ax.flatten()

        labels = ['B [G]', r'$\theta_B$', r'$\phi_B$', r'$v_\mathrm{mac}$', 'a', 'B$_0$', 'B$_1$', r'$\Delta \lambda_D$ [m$\AA$]', r'$\eta$']

        for i in range(9):
            n_pixel, n_classes = self.prob[i].shape
            x = np.linspace(self.lower[i], self.upper[i], n_classes)
            mean = np.sum(self.prob[i] * x[None,:], axis=1).reshape((512,512))

            ax[i].imshow(mean, cmap=pl.cm.viridis)
            ax[i].set_title(labels[i])

        pl.tight_layout()
        pl.show()
        # pl.savefig("{0}_{1}_comparison.png".format(self.root, self.noise))


if (__name__ == '__main__'):

    root = 'cnns/6301_hinode_1component'
    noise = 1e-4
    out = plot_map(root)
    out.read_network()
    out.forward_network()
    out.plot()