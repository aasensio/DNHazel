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

    def __init__(self, root, depth, model, activation, n_filters):

# Only allocate needed memory
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        session = tf.Session(config=config)
        ktf.set_session(session)

        self.root = root
        self.depth = depth
        self.network_type = model        
        self.activation = activation
        self.n_filters = n_filters

        self.input_file_images = "/net/viga/scratch1/cdiazbas/DATABASE/database_validation_x2.h5"
        self.input_file_images_HMI = "/net/viga/scratch1/cdiazbas/DATABASE/database_prediction.h5"

    def define_network(self, nx=50, ny=50):
        print("Setting up network...")

        self.nx = nx
        self.ny = ny

        if (self.network_type == 'encdec'):
            self.model = nn_model.encdec(self.nx, self.ny, 0.0, self.depth, activation=self.activation, n_filters=self.n_filters)

        if (self.network_type == 'keepsize'):
            self.model = nn_model.keepsize(self.nx, self.ny, 0.0, self.depth, activation=self.activation, n_filters=self.n_filters)
        
        print("Loading weights...")
        self.model.load_weights("{0}_{1}_weights.hdf5".format(self.root, self.depth))


    def predict(self):
        print("Predicting validation data...")
        tmp = np.loadtxt('/net/vena/scratch/Dropbox/GIT/DeepLearning/hmi_super/training/normalization.txt')
        self.median_HMI, self.median_SST = tmp[0], tmp[1]

        f = h5py.File(self.input_file_images_training, 'r')

        input_validation = np.zeros((1,self.nx,self.ny,12), dtype='float32')
        
        input_validation[0,:,:,:] = f[0].data[200,:,:,1:].astype('float32') / self.median

        start = time.time()
        out = self.model.predict(input_validation)
        end = time.time()
        print("Prediction took {0} seconds...".format(end-start))        
        
        print("Plotting validation data...")
        
        ff, ax = pl.subplots(nrows=2, ncols=2, figsize=(10,8))

        dat = f[0].data[200,:,:,1] / self.median
        res = ax[0,0].imshow(dat, cmap=pl.cm.gray)
        pl.colorbar(res, ax=ax[0,0])
        contrast = np.std(dat) / np.mean(dat)
        ax[0,0].set_title('contrast: {0:4.1f}%'.format(contrast*100))

        dat = f[0].data[200,:,:,2] / self.median
        res = ax[0,1].imshow(dat, cmap=pl.cm.gray)
        pl.colorbar(res, ax=ax[0,1])
        contrast = np.std(dat) / np.mean(dat)
        ax[0,1].set_title('contrast: {0:4.1f}%'.format(contrast*100))
        
        dat = f[0].data[200,10:-10,10:-10,0] / self.median
        res = ax[1,0].imshow(dat, cmap=pl.cm.gray)
        pl.colorbar(res, ax=ax[1,0])
        contrast = np.std(dat) / np.mean(dat)
        ax[1,0].set_title('contrast: {0:4.1f}%'.format(contrast*100))

        dat = out[0,10:-10,10:-10,0]
        res = ax[1,1].imshow(dat, cmap=pl.cm.gray)
        pl.colorbar(res, ax=ax[1,1])        
        contrast = np.std(dat) / np.mean(dat)
        ax[1,1].set_title('contrast: {0:4.1f}%'.format(contrast*100))

        pl.tight_layout()

        pl.show()        

        stop()

    def cube_view(self):

        axis = 0

        print("Predicting validation data...")
        tmp = np.loadtxt('/net/vena/scratch/Dropbox/GIT/DeepLearning/hmi_super/training/normalization.txt')
        self.median_HMI, self.median_SST = tmp[0], tmp[1]

        f = h5py.File(self.input_file_images, 'r')

        input_validation = np.zeros((100,self.nx,self.ny,1), dtype='float32')
        
        input_validation[:,:,:,:] = f['imHMI'][0:100,:,:,0:1].astype('float32') / self.median_HMI
        output_validation = f['imSST'][0:100,:,:,0:1].astype('float32') / self.median_SST

        start = time.time()
        out = self.model.predict(input_validation)
        end = time.time()
        print("Prediction took {0} seconds...".format(end-start))
    
        
        fig, ax = pl.subplots(nrows=1, ncols=4, figsize=(14,8))
        fig.subplots_adjust(left=0.25, bottom=0.25)

        # select first image
        s = [slice(0, 1) if i == axis else slice(None) for i in range(3)]
        im_validation = output_validation[s].squeeze()
        im_network = out[s].squeeze()
        im_original = input_validation[s].squeeze()
        
        # display image        
        l_validation = ax[0].imshow(im_validation)
        l_bilinear = ax[2].imshow(mi.imresize(im_original, 200))
        l_network = ax[3].imshow(im_network)
        l_original = ax[1].imshow(im_original)        

        ax[0].set_title('Target')
        ax[1].set_title('HMI')
        ax[2].set_title('HMI bilinear')
        ax[3].set_title('Network')

        # define slider
        axcolor = 'lightgoldenrodyellow'
        ax_bar = fig.add_axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)
        
        slider = Slider(ax_bar, 'Axis %i index' % axis, 0, output_validation.shape[axis] - 1,
                        valinit=0, valfmt='%i')

        def update(val):
            ind = int(slider.val)
            s = [slice(ind, ind + 1) if i == axis else slice(None)
                     for i in range(3)]
            im_validation = output_validation[s].squeeze()
            im_network = out[s].squeeze()
            im_original = input_validation[s].squeeze()
                        
            l_validation.set_data(im_validation)
            l_network.set_data(im_network)
            l_original.set_data(im_original)
            l_bilinear.set_data(mi.imresize(im_original, 200))
            fig.canvas.draw()

            ax[0].set_title('Target')
            ax[1].set_title('HMI')
            ax[2].set_title('HMI bilinear')
            ax[3].set_title('Network')

        slider.on_changed(update)

        pl.show()

    def gen_movie(self):

        # print("Predicting validation data...")
        # tmp = np.loadtxt('/net/vena/scratch/Dropbox/GIT/DeepLearning/hmi_super/training/normalization.txt')
        # self.median_HMI, self.median_SST = tmp[0], tmp[1]

        # f = h5py.File(self.input_file_images, 'r')

        input_validation = np.zeros((100,self.nx,self.ny,1), dtype='float32')
        
        # input_validation[:,:,:,:] = f['imHMI'][0:100,:,:,0:1].astype('float32') / self.median_HMI
        # output_validation = f['imSST'][0:100,:,:,0:1].astype('float32') / self.median_SST

        start = time.time()
        out = self.model.predict(input_validation)
        end = time.time()
        print("Prediction took {0} seconds...".format(end-start))
            
        # fig, ax = pl.subplots(nrows=1, ncols=4, figsize=(13,8))

        # n_frames = 100

        nx_up, ny_up = out[0,:,:,0].shape

        # FFMpegWriter = manimation.writers['ffmpeg']
        # metadata = dict(title='Movie Test', artist='Matplotlib', comment='Movie support!')
        # writer = FFMpegWriter(codec='libx264', fps=1, bitrate=20000, metadata=metadata, extra_args=['-pix_fmt', 'yuv420p'])
        # with writer.saving(fig, "movie.mp4", n_frames):
        #     for i in tqdm(range(n_frames)):                
                                
        #         ax[0].imshow(input_validation[i,:,:,0])
        #         ax[1].imshow(congrid.resample(input_validation[i,:,:,0], (nx_up, ny_up), minusone=True))
        #         ax[2].imshow(out[i,:,:,0])
        #         ax[3].imshow(output_validation[i,:,:,0])                
                
        #         ax[0].set_title('HMI c={0:4.1f}%'.format(contrast(input_validation[i,:,:,0])))
        #         ax[1].set_title('HMI bilinear c={0:4.1f}%'.format(contrast(congrid.resample(input_validation[i,:,:,0], (nx_up,ny_up), minusone=True))))
        #         ax[2].set_title('Network c={0:4.1f}%'.format(contrast(out[i,:,:,0])))
        #         ax[3].set_title('Target c={0:4.1f}%'.format(contrast(output_validation[i,:,:,0])))
                

        #         writer.grab_frame()
        #         ax[0].cla()
        #         ax[1].cla()
        #         ax[2].cla()  
        #         ax[3].cla() 

        print("Predicting HMI data...")        

        f = h5py.File(self.input_file_images_HMI, 'r')

        input_validation = np.zeros((100,self.nx,self.ny,1), dtype='float32')
        
        input_validation[:,:,:,:] = f['imHMI'][0:100,:,:,0:1].astype('float32')
        input_validation /= np.median(input_validation)

        start = time.time()
        out = self.model.predict(input_validation)
        end = time.time()
        print("Prediction took {0} seconds...".format(end-start))
            
        fig, ax = pl.subplots(nrows=1, ncols=3, figsize=(12,8))

        n_frames = 100

        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title='Movie Test', artist='Matplotlib', comment='Movie support!')
        writer = FFMpegWriter(codec='libx264', fps=1, bitrate=20000, metadata=metadata, extra_args=['-pix_fmt', 'yuv420p'])
        with writer.saving(fig, "movie_HMI.mp4", n_frames):
            for i in tqdm(range(n_frames)):

                v_min = np.min(out[i,:,:,0])
                v_max = np.max(out[i,:,:,0])
                                
                ax[0].imshow(input_validation[i,:,:,0], vmin=v_min, vmax=v_max) 
                ax[1].imshow(congrid.resample(input_validation[i,:,:,0], (nx_up, ny_up), minusone=True), vmin=v_min, vmax=v_max)
                ax[2].imshow(out[i,:,:,0], vmin=v_min, vmax=v_max)

                ax[0].set_title('HMI c={0:4.1f}%'.format(contrast(input_validation[i,:,:,0])))
                ax[1].set_title('HMI bilinear c={0:4.1f}%'.format(contrast(congrid.resample(input_validation[i,:,:,0], (nx_up,ny_up), minusone=True))))
                ax[2].set_title('Network c={0:4.1f}%'.format(contrast(out[i,:,:,0])))

                writer.grab_frame()
                ax[0].cla()
                ax[1].cla()
                ax[2].cla()
            
if (__name__ == '__main__'):

    parser = argparse.ArgumentParser(description='Predict for MFBD')
    parser.add_argument('-i','--input', help='Input files')
    parser.add_argument('-d','--depth', help='Depth', default=5)
    parser.add_argument('-k','--kernels', help='N. kernels', default=64)
    parser.add_argument('-m','--model', help='Model', choices=['encdec', 'keepsize'], required=True, default='keepsize')
    parser.add_argument('-c','--activation', help='Activation', choices=['relu', 'elu'], required=True, default='relu')
    parser.add_argument('-a','--action', help='action', choices=['cube', 'movie', 'large_frame'], default='cube')
    parsed = vars(parser.parse_args())

    print('Model : {0}'.format(parsed['model']))
    print('Depth : {0}'.format(parsed['depth']))
    print('Activation : {0}'.format(parsed['activation']))
    out = deep_network('../training/networks/{0}'.format(parsed['input']), depth=int(parsed['depth']), model=parsed['model'], activation=parsed['activation'],
        n_filters=int(parsed['kernels']))

    pl.close('all')
    if (parsed['action'] == 'cube'):
        out.define_network()
        out.cube_view()
    if (parsed['action'] == 'movie'):
        out.define_network()
        out.gen_movie()
    if (parsed['action'] == 'large_frame'):
        out.define_network(nx=100, ny=100)
        print("Not yet implemented")
        # out.gen_movie()

    # out.predict()