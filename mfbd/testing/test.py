import numpy as np
import matplotlib.pyplot as pl
import platform
import os
from astropy.io import fits
from ipdb import set_trace as stop
import time
import argparse
from matplotlib.widgets import Slider
import matplotlib.animation as manimation
from tqdm import tqdm

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

class deep_network(object):

    def __init__(self, root, depth, model, activation, n_filters, n_inputs):

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
        self.n_inputs = n_inputs

        self.input_file_images_training = "/net/duna/scratch1/aasensio/deepLearning/mfbd/database/8542_validation_hypercube.fits"

    def define_network(self, nx=88, ny=88):
        print("Setting up network...")

        self.nx = nx
        self.ny = ny

        if (self.network_type == 'encdec'):
            self.model = nn_model.encdec(self.nx, self.ny, 0.0, self.depth, activation=self.activation, n_filters=self.n_filters, n_inputs=self.n_inputs)

        if (self.network_type == 'keepsize'):
            self.model = nn_model.keepsize(self.nx, self.ny, 0.0, self.depth, activation=self.activation, n_filters=self.n_filters, n_inputs=self.n_inputs)
        
        print("Loading weights...")
        self.model.load_weights("{0}_{1}_weights.hdf5".format(self.root, self.depth))

    
    def cube_view(self):
        """
        View the 3D cube (x,y,frame) of original and reconstructed images
        """

        axis = 0

        print("Predicting validation data...")
        self.median = np.loadtxt('/net/duna/scratch1/aasensio/deepLearning/mfbd/database/normalizations.txt')

        f = fits.open(self.input_file_images_training, memmap=True)

        input_validation = np.zeros((100,self.nx,self.ny,self.n_inputs), dtype='float32')
        
        input_validation[:,:,:,:] = f[0].data[0:100,:,:,1:].astype('float32') / self.median
        output_validation = f[0].data[0:100,:,:,0].astype('float32') / self.median

        start = time.time()
        out = self.model.predict(input_validation)
        end = time.time()
        print("Prediction took {0} seconds...".format(end-start))
    
        
        fig, ax = pl.subplots(nrows=2, ncols=2, figsize=(10,10))        
        fig.subplots_adjust(left=0.25, bottom=0.25)

        # select first image
        s = [slice(0, 1) if i == axis else slice(None) for i in range(3)]
        im_validation = output_validation[s].squeeze()
        im_network = out[s].squeeze()
        im_original = input_validation[s].squeeze()

        # display image
        l_original1 = ax[0,0].imshow(im_original[:,:,0])
        l_original2 = ax[0,1].imshow(im_original[:,:,5])
        l_validation = ax[1,0].imshow(im_validation)
        l_network = ax[1,1].imshow(im_network)


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

            l_original1.set_data(im_original[:,:,0])
            l_original2.set_data(im_original[:,:,5])
            l_validation.set_data(im_validation)
            l_network.set_data(im_network)
            fig.canvas.draw()

        slider.on_changed(update)

        pl.show()

    def gen_movie(self):
        """
        Generate a movie of the validation frames
        """

        print("Predicting validation data...")
        self.median = np.loadtxt('/net/duna/scratch1/aasensio/deepLearning/mfbd/database/normalizations.txt')

        f = fits.open(self.input_file_images_training, memmap=True)

        input_validation = np.zeros((100,self.nx,self.ny,12), dtype='float32')
        
        input_validation[:,:,:,:] = f[0].data[0:100,:,:,1:].astype('float32') / self.median
        output_validation = f[0].data[0:100,:,:,0].astype('float32') / self.median

        start = time.time()
        out = self.model.predict(input_validation)
        end = time.time()
        print("Prediction took {0} seconds...".format(end-start))
    
        
        fig, ax = pl.subplots(nrows=2, ncols=2, figsize=(10,10))        

        n_frames = 100

        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title='Movie Test', artist='Matplotlib', comment='Movie support!')
        writer = FFMpegWriter(codec='libx264', fps=1, bitrate=20000, metadata=metadata, extra_args=['-pix_fmt', 'yuv420p'])
        with writer.saving(fig, "movie.mp4", n_frames):
            for i in tqdm(range(n_frames)):
                                
                ax[0,0].imshow(input_validation[i,:,:,0])
                ax[0,1].imshow(input_validation[i,:,:,5])
                ax[1,0].imshow(output_validation[i,:,:])
                ax[1,1].imshow(out[i,:,:,0])        

                writer.grab_frame()
                ax[0,0].cla()
                ax[0,1].cla()
                ax[1,0].cla()
                ax[1,1].cla()

    def gen_movie_allcams(self):
        """
        Generate a movie of the validation data for several cameras to see if the network is generalizing correctly
        """

        print("Predicting validation data...")
        self.median = np.loadtxt('/net/duna/scratch1/aasensio/deepLearning/mfbd/database/normalizations.txt')

        f_1 = fits.open('/net/duna/scratch1/aasensio/deepLearning/mfbd/testing/camXIX_validating_hypercube.fits', memmap=True)
        f_2 = fits.open('/net/duna/scratch1/aasensio/deepLearning/mfbd/testing/camXX_validating_hypercube.fits', memmap=True)
        f_3 = fits.open('/net/duna/scratch1/aasensio/deepLearning/mfbd/testing/camXXV_validating_hypercube.fits', memmap=True)

        input_validation = np.zeros((100,self.nx,self.ny,12), dtype='float32')
        
        
        output_validation_1 = f_1[0].data[0:100,:,:,0].astype('float32') / self.median
        output_validation_2 = f_2[0].data[0:100,:,:,0].astype('float32') / self.median
        output_validation_3 = f_3[0].data[0:100,:,:,0].astype('float32') / self.median

        print("Camera XIX...")
        input_validation[:,:,:,:] = f_1[0].data[0:100,:,:,1:].astype('float32') / self.median
        start = time.time()
        out_1 = self.model.predict(input_validation)
        end = time.time()
        print("Prediction took {0} seconds...".format(end-start))

        print("Camera XX...")
        input_validation[:,:,:,:] = f_2[0].data[0:100,:,:,1:].astype('float32') / self.median
        start = time.time()
        out_2 = self.model.predict(input_validation)
        end = time.time()
        print("Prediction took {0} seconds...".format(end-start))

        print("Camera XXV...")
        input_validation[:,:,:,:] = f_3[0].data[0:100,:,:,1:].astype('float32') / self.median
        start = time.time()
        out_3 = self.model.predict(input_validation)
        end = time.time()
        print("Prediction took {0} seconds...".format(end-start))
            
        fig, ax = pl.subplots(nrows=2, ncols=3, figsize=(12,10))

        n_frames = 100

        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title='Movie Test', artist='Matplotlib', comment='Movie support!')
        writer = FFMpegWriter(codec='libx264', fps=1, bitrate=20000, metadata=metadata, extra_args=['-pix_fmt', 'yuv420p'])
        with writer.saving(fig, "movie_allcams_{0}_{1}.mp4".format(self.network_type,self.depth), n_frames):
            for i in tqdm(range(n_frames)):
                                
                ax[0,0].imshow(out_1[i,:,:,0])
                ax[0,1].imshow(out_2[i,:,:,0])
                ax[0,2].imshow(out_3[i,:,:,0])

                ax[1,0].imshow(output_validation_1[i,:,:])
                ax[1,1].imshow(output_validation_2[i,:,:])
                ax[1,2].imshow(output_validation_3[i,:,:])

                ax[0,0].set_title('Camera XIX')
                ax[0,1].set_title('Camera XX')
                ax[0,2].set_title('Camera XXV')

                ax[0,0].set_ylabel('Output network')
                ax[1,0].set_ylabel('MOMFBD')
                
                writer.grab_frame()
                ax[0,0].cla()
                ax[0,1].cla()
                ax[0,2].cla()
                ax[1,0].cla()
                ax[1,1].cla()
                ax[1,2].cla()

    def gen_large_frame(self):
        """
        Apply the network to a large FOV, both by subframes and to the whole large frame
        """

        print("Predicting validation data...")
        self.median = np.loadtxt('/net/duna/scratch1/aasensio/deepLearning/mfbd/database/normalizations.txt')

        f_1 = fits.open('/net/duna/scratch1/aasensio/deepLearning/mfbd/testing/camXIX_validating_hypercube_single.fits', memmap=True)
        f_2 = fits.open('/net/duna/scratch1/aasensio/deepLearning/mfbd/testing/camXX_validating_hypercube_single.fits', memmap=True)
        f_3 = fits.open('/net/duna/scratch1/aasensio/deepLearning/mfbd/testing/camXXV_validating_hypercube_single.fits', memmap=True)

        input_validation = np.zeros((1,self.nx,self.ny,12), dtype='float32')
        
        output_validation_1 = f_1[0].data[0,:,:,0].astype('float32') / self.median
        output_validation_2 = f_2[0].data[0,:,:,0].astype('float32') / self.median
        output_validation_3 = f_3[0].data[0,:,:,0].astype('float32') / self.median

        print("Camera XIX...")
        input_validation[:,:,:,:] = f_1[0].data[0:1,:,:,1:].astype('float32') / self.median
        start = time.time()
        out_1 = self.model.predict(input_validation)
        end = time.time()
        print("Prediction took {0} seconds...".format(end-start))

        print("Camera XX...")
        input_validation[:,:,:,:] = f_2[0].data[0:1,:,:,1:].astype('float32') / self.median
        start = time.time()
        out_2 = self.model.predict(input_validation)
        end = time.time()
        print("Prediction took {0} seconds...".format(end-start))

        print("Camera XXV...")
        input_validation[:,:,:,:] = f_3[0].data[0:1,:,:,1:].astype('float32') / self.median
        start = time.time()
        out_3 = self.model.predict(input_validation)
        end = time.time()
        print("Prediction took {0} seconds...".format(end-start))
            
        fig, ax = pl.subplots(nrows=2, ncols=3, figsize=(12,10))
        
        ax[0,0].imshow(np.clip(out_1[0,:,:,0], 0, None))
        ax[0,1].imshow(np.clip(out_2[0,:,:,0], 0, None))
        ax[0,2].imshow(np.clip(out_3[0,:,:,0], 0, None))

        ax[1,0].imshow(output_validation_1[:,:])
        ax[1,1].imshow(output_validation_2[:,:])
        ax[1,2].imshow(output_validation_3[:,:])

        ax[0,0].set_title('Camera XIX')
        ax[0,1].set_title('Camera XX')
        ax[0,2].set_title('Camera XXV')

        ax[0,0].set_ylabel('Output network')
        ax[1,0].set_ylabel('MOMFBD')

        stop()

        pl.show()

        pl.savefig('large_frame.png')
                                    
if (__name__ == '__main__'):

    parser = argparse.ArgumentParser(description='Predict for MFBD')
    parser.add_argument('-i','--input', help='Input files')
    parser.add_argument('-d','--depth', help='Depth', default=5)
    parser.add_argument('-k','--kernels', help='N. kernels', default=64)
    parser.add_argument('-m','--model', help='Model', choices=['encdec', 'keepsize'], required=True)
    parser.add_argument('-c','--activation', help='Activation', choices=['relu', 'elu'], required=True, default='relu')
    parser.add_argument('-ni','--ninputs', help='N. inputs', required=True, default=12)
    parser.add_argument('-a','--action', help='action', choices=['cube', 'movie', 'movie_multi', 'large_frame'], default='cube')
    parsed = vars(parser.parse_args())

    print('Model : {0}'.format(parsed['model']))
    print('Depth : {0}'.format(parsed['depth']))
    print('Activation : {0}'.format(parsed['activation']))
    out = deep_network('../training/networks/{0}'.format(parsed['input']), depth=int(parsed['depth']), model=parsed['model'], activation=parsed['activation'],
        n_filters=int(parsed['kernels']), n_inputs=int(parsed['ninputs']))

    pl.close('all')
    if (parsed['action'] == 'cube'):
        out.define_network()
        out.cube_view()
    if (parsed['action'] == 'movie'):
        out.define_network()
        out.gen_movie()
    if (parsed['action'] == 'movie_multi'):
        out.define_network()
        out.gen_movie_allcams()
    if (parsed['action'] == 'large_frame'):
        out.define_network(nx=974, ny=956)
        out.gen_large_frame()
    # out.predict()
