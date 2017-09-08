import numpy as np
import matplotlib.pyplot as pl
import platform
import os
from ipdb import set_trace as stop
from astropy.io import fits
import scipy.io
import time

if (platform.node() == 'viga'):
    os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32"

os.environ["KERAS_BACKEND"] = "tensorflow"

if (platform.node() != 'viga'):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
import keras.backend.tensorflow_backend as ktf

import sys
sys.path.append('../training')

import encdec_model as nn_model

class phasediv(object):

    def __init__(self, root, depth):

# Only allocate needed memory
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        session = tf.Session(config=config)
        ktf.set_session(session)

        self.root = root
        self.nx = 700
        self.ny = 700
        self.depth = depth

    def define_network(self):
        print("Setting up network...")

        self.model = nn_model.define_network(self.nx, self.ny, noise=0.0, depth=self.depth)

        print("Loading weights...")
        self.model.load_weights("{0}_{1}_weights.hdf5".format(self.root, self.depth))


    def validation_generator_1stflight(self):
        f_focus = fits.open('imax_focus.fits')
        f_defocus = fits.open('imax_defocus.fits')

        im_focus = f_focus[0].data
        im_defocus = f_defocus[0].data        
        
        self.median = np.median(im_focus[100:-100,100:-100])

        input_validation = np.zeros((self.batch_size,self.nx,self.ny,2), dtype='float32')

        while 1:        
            for i in range(1):

                input_validation[i*self.batch_size:(i+1)*self.batch_size,:,:,0] = im_focus[100:100+self.nx,100:100+self.ny] / self.median
                input_validation[i*self.batch_size:(i+1)*self.batch_size,:,:,1] = im_defocus[100:100+self.nx,100:100+self.ny] / self.median

                yield input_validation

        f_focus.close()
        f_defocus.close()

    def predict_validation_1stflight(self):
        print("Predicting validation data...")

        f_focus = fits.open('imax_focus.fits')
        f_defocus = fits.open('imax_defocus.fits')

        im_focus = f_focus[0].data
        im_defocus = f_defocus[0].data

        f_estimated = fits.open('imaxf_image_estimated.fits')
        im_estimated = f_estimated[0].data

        # start = time.time()
        # out = self.model.predict_generator(self.validation_generator_1stflight(), self.batch_size)

        input_validation = np.zeros((1,self.nx,self.ny,2), dtype='float32')
        self.median = np.median(im_focus[100:-100,100:-100])
        input_validation[0,:,:,0] = im_focus[100:100+self.nx,100:100+self.ny] / self.median
        input_validation[0,:,:,1] = im_defocus[100:100+self.nx,100:100+self.ny] / self.median

        start = time.time()
        out = self.model.predict(input_validation)
        end = time.time()
        print("Prediction took {0} seconds...".format(end-start))        

        print("Plotting validation data...")

        
        f, ax = pl.subplots(nrows=2, ncols=2, figsize=(15,10))
        
        res = ax[0,0].imshow(im_focus[100:100+self.nx,100:100+self.ny], cmap=pl.cm.gray)
        pl.colorbar(res, ax=ax[0,0])
        dat = im_focus[100:100+self.nx,100:100+self.ny]
        contrast = np.std(dat) / np.mean(dat)
        ax[0,0].set_title('contrast: {0:4.1f}%'.format(contrast*100))

        res = ax[0,1].imshow(im_defocus[100:100+self.nx,100:100+self.ny], cmap=pl.cm.gray)
        pl.colorbar(res, ax=ax[0,1])
        dat = im_defocus[100:100+self.nx,100:100+self.ny]
        contrast = np.std(dat) / np.mean(dat)
        ax[0,1].set_title('contrast: {0:4.1f}%'.format(contrast*100))

        res = ax[1,0].imshow(out[0,:,:,0] * self.median, cmap=pl.cm.gray)
        pl.colorbar(res, ax=ax[1,0])
        dat = out[0,:,:,0] * self.median
        contrast = np.std(dat) / np.mean(dat)
        ax[1,0].set_title('contrast: {0:4.1f}%'.format(contrast*100))

        res = ax[1,1].imshow(im_estimated[100:100+self.nx,100:100+self.ny], cmap=pl.cm.gray)
        pl.colorbar(res, ax=ax[1,1])
        dat = im_estimated[100:100+self.nx,100:100+self.ny]
        contrast = np.std(dat) / np.mean(dat)
        ax[1,1].set_title('contrast: {0:4.1f}%'.format(contrast*100))

        pl.tight_layout()

        pl.show()        

        pl.savefig("{0}_imax.png".format(self.root))


    def validation_generator_2ndflight(self):
        f = fits.open('/net/vena/scratch1/deepLearning/phaseDiv/imax/12-06-2013/pd.001.fits')
        im_focus = f[0].data[:,0:936]
        im_defocus = f[0].data[:,936:]
        
        self.median = np.median(im_focus[100:-100,100:-100])

        input_validation = np.zeros((self.batch_size,self.nx,self.ny,2), dtype='float32')

        while 1:        
            for i in range(1):

                input_validation[i*self.batch_size:(i+1)*self.batch_size,:,:,0] = im_focus[100:100+self.nx,100:100+self.ny] / self.median
                input_validation[i*self.batch_size:(i+1)*self.batch_size,:,:,1] = im_defocus[100:100+self.nx,100:100+self.ny] / self.median

                yield input_validation

        f.close()

    def predict_validation_2ndflight(self):
        print("Predicting validation data...")

        f = fits.open('/net/vena/scratch1/deepLearning/phaseDiv/imax/12-06-2013/pd.001.fits')
        im_focus = f[0].data[:,0:936]
        im_defocus = f[0].data[:,936:]
        
        f_estimated = scipy.io.readsav('/net/vena/scratch1/deepLearning/phaseDiv/imax/r_pd_nodefoc.001.save')
        im_estimated = f_estimated['grandscene']

        input_validation = np.zeros((1,self.nx,self.ny,2), dtype='float32')
        self.median = np.median(im_focus[100:-100,100:-100])
        input_validation[0,:,:,0] = im_focus[100:100+self.nx,100:100+self.ny] / self.median
        input_validation[0,:,:,1] = im_defocus[100:100+self.nx,100:100+self.ny] / self.median

        start = time.time()
        out = self.model.predict(input_validation)
        end = time.time()
        print("Prediction took {0} seconds...".format(end-start))

        # start = time.time()
        # out = self.model.predict_generator(self.validation_generator_2ndflight(), self.batch_size)
        # end = time.time()
        # print("Prediction took {0} seconds...".format(end-start))

        print("Plotting validation data...")

        f, ax = pl.subplots(nrows=2, ncols=2, figsize=(15,10))
        
        res = ax[0,0].imshow(im_focus[100:100+self.nx,100:100+self.ny], cmap=pl.cm.gray)
        pl.colorbar(res, ax=ax[0,0])
        dat = im_focus[100:100+self.nx,100:100+self.ny]
        contrast = np.std(dat) / np.mean(dat)
        ax[0,0].set_title('contrast: {0:4.1f}%'.format(contrast*100))

        res = ax[0,1].imshow(im_defocus[100:100+self.nx,100:100+self.ny], cmap=pl.cm.gray)
        pl.colorbar(res, ax=ax[0,1])
        dat = im_defocus[100:100+self.nx,100:100+self.ny]
        contrast = np.std(dat) / np.mean(dat)
        ax[0,1].set_title('contrast: {0:4.1f}%'.format(contrast*100))

        res = ax[1,0].imshow(out[0,:,:,0] * self.median, cmap=pl.cm.gray)
        pl.colorbar(res, ax=ax[1,0])
        dat = out[0,:,:,0] * self.median
        contrast = np.std(dat) / np.mean(dat)
        ax[1,0].set_title('contrast: {0:4.1f}%'.format(contrast*100))

        res = ax[1,1].imshow(im_estimated[100:100+self.nx,100:100+self.ny], cmap=pl.cm.gray)
        pl.colorbar(res, ax=ax[1,1])
        dat = im_estimated[100:100+self.nx,100:100+self.ny]
        contrast = np.std(dat) / np.mean(dat)
        ax[1,1].set_title('contrast: {0:4.1f}%'.format(contrast*100))

        pl.tight_layout()

        pl.show()        

        # pl.savefig("{0}_imax.png".format(self.root))
            
if (__name__ == '__main__'):
    
    out = phasediv('../training/networks/encdec', depth=8)
    out.define_network()

    pl.close('all')
    out.predict_validation_1stflight()
    # out.predict_validation_2ndflight()
