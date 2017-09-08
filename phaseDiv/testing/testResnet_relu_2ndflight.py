import numpy as np
import matplotlib.pyplot as pl
import h5py
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

from keras.layers import Input, Convolution2D, merge, Activation, Lambda, BatchNormalization
from keras.callbacks import ModelCheckpoint, Callback
from keras.models import Model, model_from_json
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / N 

class trainDNNFull(object):

    def __init__(self, root):

# Only allocate needed memory
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        session = tf.Session(config=config)
        ktf.set_session(session)

        self.root = root
        self.nx = 700
        self.ny = 700
        self.n_times = 2
        self.n_filters = 64
        self.batch_size = 1      
        self.n_conv_layers = 16
        self.stride = 1
        self.skip_frequency = 2
        self.n_diversity = 2
        self.input_file_images_validation = "/net/duna/scratch1/aasensio/deepLearning/phaseDiv/database/database_images_validation.h5"        

    def residual(self, inputs):
        x = Convolution2D(self.n_filters, 3, 3, border_mode='same', init='he_normal')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Convolution2D(self.n_filters, 3, 3, border_mode='same', init='he_normal')(x)
        x = BatchNormalization()(x)
        x = merge([x, inputs], 'sum')

        return x    
            
    def defineNetwork(self):
        print("Setting up network...")

        inputs = Input(shape=(self.nx, self.ny, self.n_diversity))
        conv = Convolution2D(self.n_filters, 3, 3, activation='relu', border_mode='same', init='he_normal')(inputs)

        x = self.residual(conv)
        for i in range(self.n_conv_layers):
            x = self.residual(x)

        x = Convolution2D(self.n_filters, 3, 3, border_mode='same', init='he_normal')(x)
        x = BatchNormalization()(x)
        x = merge([x, conv], 'sum')

        final = Convolution2D(1, 1, 1, activation='relu', border_mode='same', init='he_normal')(x)

        self.model = Model(input=inputs, output=final)

        print("Loading weights...")
        self.model.load_weights("{0}_weights.hdf5".format(self.root))

    def validation_generator(self):
        f = fits.open('/net/vena/scratch1/deepLearning/phaseDiv/imax/12-06-2013/pd.001.fits')
        im_focus = f[0].data[:,0:936]
        im_defocus = f[0].data[:,936:]
        
        self.min_i = np.min(im_focus[100:-100,100:-100])
        self.max_i = np.max(im_focus[100:-100,100:-100])

        input_validation = np.zeros((self.batch_size,self.nx,self.ny,2), dtype='float32')

        while 1:        
            for i in range(1):

                input_validation[i*self.batch_size:(i+1)*self.batch_size,:,:,0] = (im_focus[100:100+self.nx,100:100+self.ny] - self.min_i) / (self.max_i - self.min_i)
                input_validation[i*self.batch_size:(i+1)*self.batch_size,:,:,1] = (im_defocus[100:100+self.nx,100:100+self.ny] - self.min_i) / (self.max_i - self.min_i)

                yield input_validation

        f.close()        

    def predict_validation(self):
        print("Predicting validation data...")

        f = fits.open('/net/vena/scratch1/deepLearning/phaseDiv/imax/12-06-2013/pd.001.fits')
        im_focus = f[0].data[:,0:936]
        im_defocus = f[0].data[:,936:]

        f_estimated = scipy.io.readsav('/net/vena/scratch1/deepLearning/phaseDiv/imax/r_pd_nodefoc.001.save')
        im_estimated = f_estimated['grandscene']

        start = time.time()
        out = self.model.predict_generator(self.validation_generator(), self.batch_size)
        end = time.time()
        print("Prediction took {0} seconds...".format(end-start))

        print("Plotting validation data...")

        f.close()

        
        pl.close('all')   

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

        res = ax[1,0].imshow(out[0,:,:,0] * (self.max_i - self.min_i) + self.min_i, cmap=pl.cm.gray)
        pl.colorbar(res, ax=ax[1,0])
        dat = out[0,:,:,0] * (self.max_i - self.min_i) + self.min_i
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



        stop()
            
if (__name__ == '__main__'):
    
    out = trainDNNFull('../training/cnns/resnet_noise1e-3')
    out.defineNetwork()
    out.predict_validation()
