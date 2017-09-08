import numpy as np
import matplotlib.pyplot as pl
import platform
import os
from astropy.io import fits
from ipdb import set_trace as stop
from astropy.io import fits
import scipy.io
import time
import argparse
import h5py
from matplotlib.widgets import Slider
import matplotlib.animation as manimation
from tqdm import tqdm
import scipy.misc as mi
import congrid

if (platform.node() == 'viga'):
    os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32"

os.environ["KERAS_BACKEND"] = "tensorflow"

if (platform.node() != 'viga'):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
import keras.backend.tensorflow_backend as ktf

import sys
sys.path.append('../training')

import models as nn_model

def contrast(x):
    return 100 * np.std(x) / np.mean(x)

class deep_network(object):

    def __init__(self, networks, depths, filters_exit, activation, n_filters):

# Only allocate needed memory
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        session = tf.Session(config=config)
        ktf.set_session(session)

        self.networks = networks
        self.activation = activation
        self.n_filters = n_filters
        self.depth = depths
        self.filters_exit = filters_exit


    def define_network(self, root, depth, filters_exit=256, nx=50, ny=50):
        print("Setting up network...")

        if (filters_exit == 256):
            self.model = nn_model.keepsize_256(self.nx, self.ny, 0.0, depth, activation=self.activation, n_filters=self.n_filters)
        else:
            self.model = nn_model.keepsize(self.nx, self.ny, 0.0, depth, activation=self.activation, n_filters=self.n_filters)
        
        print("Loading weights...")
        self.model.load_weights("../training/networks/{0}_{1}_weights.hdf5".format(root, depth))
    
    def gen_movie(self, nx=50, ny=50):

        self.nx = nx
        self.ny = ny

        self.input_file_images_HMI = "/net/viga/scratch1/cdiazbas/DATABASE/database_prediction.h5"

        print("Predicting HMI data...")        

        f = h5py.File(self.input_file_images_HMI, 'r')

        input_validation = np.zeros((100,self.nx,self.ny,1), dtype='float32')
        
        input_validation[:,:,:,:] = f['imHMI'][0:100,:,:,0:1].astype('float32')
        input_validation /= np.median(input_validation)

        out = [None] * len(self.networks)

        for i in range(len(self.networks)):
            self.define_network(self.networks[i], self.depth[i], self.filters_exit[i])
            start = time.time()
            out[i] = self.model.predict(input_validation)
            end = time.time()
            print("Prediction took {0} seconds...".format(end-start))

        nx_up, ny_up = out[0][0,:,:,0].shape

        fig, ax = pl.subplots(nrows=2, ncols=3, figsize=(12,10))

        ax = ax.flatten()

        n_frames = 100

        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title='Movie Test', artist='Matplotlib', comment='Movie support!')
        writer = FFMpegWriter(codec='libx264', fps=1, bitrate=20000, metadata=metadata, extra_args=['-pix_fmt', 'yuv420p'])
        with writer.saving(fig, "movie_HMI_alldepths.mp4", n_frames):
            for i in tqdm(range(n_frames)):

                v_min = np.min(out[0][i,:,:,0])
                v_max = np.max(out[0][i,:,:,0])
                                
                ax[0].imshow(input_validation[i,:,:,0], vmin=v_min, vmax=v_max) 
                ax[1].imshow(congrid.resample(input_validation[i,:,:,0], (nx_up, ny_up), minusone=True), vmin=v_min, vmax=v_max)
                ax[2].imshow(out[0][i,:,:,0], vmin=v_min, vmax=v_max)
                ax[3].imshow(out[1][i,:,:,0], vmin=v_min, vmax=v_max)
                ax[4].imshow(out[2][i,:,:,0], vmin=v_min, vmax=v_max)
                ax[5].imshow(out[3][i,:,:,0], vmin=v_min, vmax=v_max)

                ax[0].set_title('HMI c={0:4.1f}%'.format(contrast(input_validation[i,:,:,0])))
                ax[1].set_title('HMI bilinear c={0:4.1f}%'.format(contrast(congrid.resample(input_validation[i,:,:,0], (nx_up,ny_up), minusone=True))))
                ax[2].set_title('Network Depth 15 c={0:4.1f}%'.format(contrast(out[0][i,:,:,0])))
                ax[3].set_title('Network Depth 5 c={0:4.1f}%'.format(contrast(out[1][i,:,:,0])))
                ax[4].set_title('Network Depth 5 64 kernels at exit c={0:4.1f}%'.format(contrast(out[2][i,:,:,0])))
                ax[5].set_title('Network Depth 5 PSF2 c={0:4.1f}%'.format(contrast(out[3][i,:,:,0])))

                writer.grab_frame()
                ax[0].cla()
                ax[1].cla()
                ax[2].cla()
                ax[3].cla()
                ax[4].cla()
                ax[5].cla()


    def gen_movie_blos_validation(self, nx=50, ny=50):

        self.nx = nx
        self.ny = ny

        print("Predicting validation data...")
        self.median_HMI, self.median_SST = 1.0, 1.0
        self.input_file_images = "/net/viga/scratch1/cdiazbas/DATABASE/database_validation_x2_BLOS.h5"

        f = h5py.File(self.input_file_images, 'r')

        input_validation = np.zeros((100,self.nx,self.ny,1), dtype='float32')
        
        input_validation[:,:,:,:] = f['imHMI'][0:100,:,:,0:1].astype('float32') / self.median_HMI
        output_validation = f['imSST'][0:100,:,:,0:1].astype('float32') / self.median_SST

        out = [None] * len(self.networks)

        for i in range(len(self.networks)):
            self.define_network(self.networks[i], self.depth[i], self.filters_exit[i])
            start = time.time()
            out[i] = self.model.predict(input_validation)
            end = time.time()
            print("Prediction took {0} seconds...".format(end-start))
            
        fig, ax = pl.subplots(nrows=1, ncols=4, figsize=(13,8))

        n_frames = 100

        nx_up, ny_up = out[0][0,:,:,0].shape

        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title='Movie Test', artist='Matplotlib', comment='Movie support!')
        writer = FFMpegWriter(codec='libx264', fps=1, bitrate=20000, metadata=metadata, extra_args=['-pix_fmt', 'yuv420p'])
        with writer.saving(fig, "movie_blos.mp4", n_frames):
            for i in tqdm(range(n_frames)):                
                                
                ax[0].imshow(input_validation[i,:,:,0])
                ax[1].imshow(congrid.resample(input_validation[i,:,:,0], (nx_up, ny_up), minusone=True))
                ax[2].imshow(out[0][i,:,:,0])
                ax[3].imshow(output_validation[i,:,:,0])                
                
                ax[0].set_title('HMI c={0:4.1f}%'.format(contrast(input_validation[i,:,:,0])))
                ax[1].set_title('HMI bilinear c={0:4.1f}%'.format(contrast(congrid.resample(input_validation[i,:,:,0], (nx_up,ny_up), minusone=True))))
                ax[2].set_title('Network c={0:4.1f}%'.format(contrast(out[0][i,:,:,0])))
                ax[3].set_title('Target c={0:4.1f}%'.format(contrast(output_validation[i,:,:,0])))
                

                writer.grab_frame()
                ax[0].cla()
                ax[1].cla()
                ax[2].cla()  
                ax[3].cla() 

    def gen_movie_blos(self, nx=50, ny=50):

        self.nx = nx
        self.ny = ny

        self.input_file_images_HMI = "/net/viga/scratch1/cdiazbas/DATABASE/database_prediction_BLOS.h5"

        print("Predicting HMI data...")        

        f = h5py.File(self.input_file_images_HMI, 'r')

        input_validation = np.zeros((100,self.nx,self.ny,1), dtype='float32')
        
        input_validation[:,:,:,:] = f['imHMI'][0:100,:,:,0:1].astype('float32')

        out = [None] * len(self.networks)

        for i in range(len(self.networks)):
            self.define_network(self.networks[i], self.depth[i], self.filters_exit[i])
            start = time.time()
            out[i] = self.model.predict(input_validation)
            end = time.time()
            print("Prediction took {0} seconds...".format(end-start))

        nx_up, ny_up = out[0][0,:,:,0].shape

        fig, ax = pl.subplots(nrows=1, ncols=4, figsize=(14,7))

        ax = ax.flatten()

        n_frames = 100

        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title='Movie Test', artist='Matplotlib', comment='Movie support!')
        writer = FFMpegWriter(codec='libx264', fps=1, bitrate=20000, metadata=metadata, extra_args=['-pix_fmt', 'yuv420p'])
        with writer.saving(fig, "movie_HMI_blos.mp4", n_frames):
            for i in tqdm(range(n_frames)):

                v_min = np.min(out[0][i,:,:,0])
                v_max = np.max(out[0][i,:,:,0])
                                
                ax[0].imshow(input_validation[i,:,:,0], vmin=v_min, vmax=v_max) 
                ax[1].imshow(congrid.resample(input_validation[i,:,:,0], (nx_up, ny_up), minusone=True), vmin=v_min, vmax=v_max)
                ax[2].imshow(out[0][i,:,:,0], vmin=v_min, vmax=v_max)
                ax[3].imshow(out[1][i,:,:,0], vmin=v_min, vmax=v_max)
                
                ax[0].set_title('HMI')
                ax[1].set_title('HMI bilinear')
                ax[2].set_title('Network Depth 5')
                ax[3].set_title('Network Depth 5 regularized')

                writer.grab_frame()
                ax[0].cla()
                ax[1].cla()
                ax[2].cla()
                ax[3].cla()
            
if (__name__ == '__main__'):

    pl.close('all')

    networks = ['keepsize_x2_blos', 'keepsize_x2_blos_2']
    depths = [5, 5]
    filters_exit = [64, 64]
    out = deep_network(networks, depths, filters_exit, activation='relu', n_filters=64)
    # out.gen_movie_blos_validation()
    out.gen_movie_blos()


    # networks = ['keepsize_x2_PSF1', 'keepsize_x2_PSF1', 'keepsize_x2_PSF2_64out', 'keepsize_x2_PSF2']
    # depths = [15,10,5,5,5]
    # filters_exit = [256, 256, 64, 256]
    # out = deep_network(networks, depths, filters_exit, activation='relu', n_filters=64)
    # out.gen_movie()

