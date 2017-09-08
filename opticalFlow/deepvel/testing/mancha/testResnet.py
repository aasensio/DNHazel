import numpy as np
import matplotlib.pyplot as pl
import h5py
import platform
import os
from ipdb import set_trace as stop
from astropy.io import fits
import scipy.io as io
import time
import matplotlib.animation as manimation

os.environ["KERAS_BACKEND"] = "tensorflow"

if (platform.node() != 'vena'):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from keras.layers import Input, Convolution2D, merge, Activation, Lambda, BatchNormalization
from keras.callbacks import ModelCheckpoint, Callback
from keras.models import Model, model_from_json
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf

import poppy
import pyfftw
import scipy.special as sp
from astropy import units as u
import congrid


def _even(x):
    return x%2 == 0

def _zernike_parity(j, jp):
    return _even(j-jp)

def _zernike_coeff_kolmogorov(D, r0, n_zernike):
    """
    Return Zernike coefficients in phase units
    """
    covariance = np.zeros((n_zernike,n_zernike))
    for j in range(n_zernike):
        n, m = poppy.zernike.noll_indices(j+1)
        for jpr in range(n_zernike):
            npr, mpr = poppy.zernike.noll_indices(jpr+1)
            
            deltaz = (m == mpr) and (_zernike_parity(j, jpr) or m == 0)
            
            if (deltaz):                
                phase = (-1.0)**(0.5*(n+npr-2*m))
                t1 = np.sqrt((n+1)*(npr+1)) 
                t2 = sp.gamma(14./3.0) * sp.gamma(11./6.0)**2 * (24.0/5.0*sp.gamma(6.0/5.0))**(5.0/6.0) / (2.0*np.pi**2)

                Kzz = t2 * t1 * phase
                
                t1 = sp.gamma(0.5*(n+npr-5.0/3.0)) * (D / r0)**(5.0/3.0)
                t2 = sp.gamma(0.5*(n-npr+17.0/3.0)) * sp.gamma(0.5*(npr-n+17.0/3.0)) * sp.gamma(0.5*(n+npr+23.0/3.0))
                covariance[j,jpr] = Kzz * t1 / t2


    covariance[0,0] = 1.0

    out = np.random.multivariate_normal(np.zeros(n_zernike), covariance)

    out[0] = 0.0

    return out

class imax_degradation(object):

    def __init__(self, telescope_radius, pixel_size, fov, secondary_radius=None):
        self.telescope_radius = telescope_radius
        if (secondary_radius != None):
            self.secondary_radius = secondary_radius
        else:
            self.secondary_radius = 0.0
        self.pixel_size = pixel_size
        self.fov = fov
        self.zernike_max = 45
        self.r0 = 10.0 * u.cm

    def compute_psf(self, lambda0, rms_telescope=1.0/9.0, rms_atmosphere=1.0/9.0):
        self.lambda0 = lambda0

        osys = poppy.OpticalSystem()

        osys.add_pupil(poppy.CircularAperture(radius = self.telescope_radius))
        
        if (self.secondary_radius != 0):
            osys.add_pupil(poppy.SecondaryObscuration(secondary_radius = self.secondary_radius))

# Telescope diffraction + aberrations
        phase_telescope = np.random.randn(self.zernike_max)
        sigma = np.sqrt(np.sum(phase_telescope[4:]**2)) / 2.0 / np.pi
        phase_telescope *= rms_telescope / sigma
        phase_telescope[0:4] = 0.0
        
# Atmosphere
        phase_atmosphere = _zernike_coeff_kolmogorov(2.0 * self.telescope_radius.to(u.cm).value, self.r0.to(u.cm).value, self.zernike_max)
        sigma = np.sqrt(np.sum(phase_atmosphere[4:]**2)) / 2.0 / np.pi
        phase_atmosphere *= rms_atmosphere / sigma
        phase_atmosphere[0:4] = 0.0

        thinlens = poppy.ZernikeWFE(radius=self.telescope_radius.to(u.m).value, coefficients=(phase_telescope + phase_atmosphere) * lambda0.to(u.m).value / (2.0 * np.pi))
        osys.add_pupil(thinlens)

        osys.add_detector(pixelscale=self.pixel_size, fov_pixels=self.fov, oversample=1)

        psf = osys.calc_psf(wavelength=self.lambda0)

        self.psf = psf[0].data
        nx_psf, ny_psf = psf[0].data.shape

        psf = np.roll(self.psf, int(nx_psf/2), axis=0)
        psf = np.roll(psf, int(ny_psf/2), axis=1)

        self.psf_fft = pyfftw.interfaces.numpy_fft.fft2(psf)

    def apply_psf(self, image):
        image_fft = pyfftw.interfaces.numpy_fft.fft2(image)
        self.image = np.real(pyfftw.interfaces.numpy_fft.ifft2(self.psf_fft * image_fft))
        return self.image

    def rebin_image(self, nx, ny):
        return congrid.resample(self.image, (nx, ny))


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / N 

class trainDNNFull(object):

    def __init__(self, root, output, name_of_variable):

# Only allocate needed memory
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        session = tf.Session(config=config)
        ktf.set_session(session)

        self.root = root
        self.nx = 576
        self.ny = 576
        self.n_times = 2
        self.n_filters = 64
        self.batch_size = 1
        self.n_conv_layers = 20
        self.stride = 1
        self.skip_frequency = 2
        self.n_frames = 1        
        self.output = output
        self.name_of_variable = name_of_variable

        telescope_radius = 0.5 * 0.965 * u.meter
        pixel_size = 0.02759 * u.arcsec / u.pixel
        fov = 1152 * u.pixel
        lambda0 = 500 * u.nm
        imax = imax_degradation(telescope_radius, pixel_size, fov)
        imax.compute_psf(lambda0)


        res = io.readsav('/net/viga/scratch1/deepLearning/opticalFlow/mancha/c3d_1152_cont4_4bin_012000_continuum.sav')['continuum']

        self.images = np.zeros((2,576,576), dtype='float32')

# 576 pixels are obtained by resampling 1152 pixels of 0.02759 "/px to 0.0545 "/px for IMaX
        self.images[0,:,:] = congrid.resample(imax.apply_psf(res[0,:,:]), (576, 576))
        self.images[1,:,:] = congrid.resample(imax.apply_psf(res[1,:,:]), (576, 576))

        res = io.readsav('/net/viga/scratch1/deepLearning/opticalFlow/mancha/c3d_1152_cont4_4bin_012000.isotau.sav')

        self.vx = np.zeros((3,576,576), dtype='float32')
        self.vy = np.zeros((3,576,576), dtype='float32')

        for i in range(3):
            self.vx[i,:,:] = congrid.resample(imax.apply_psf(res['vx'][i,:,:]), (576, 576))
            self.vy[i,:,:] = congrid.resample(imax.apply_psf(res['vy'][i,:,:]), (576, 576))
        
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

        inputs = Input(shape=(self.nx, self.ny, self.n_times))
        conv = Convolution2D(self.n_filters, 3, 3, activation='relu', border_mode='same', init='he_normal')(inputs)

        x = self.residual(conv)
        for i in range(self.n_conv_layers):
            x = self.residual(x)

        x = Convolution2D(self.n_filters, 3, 3, border_mode='same', init='he_normal')(x)
        x = BatchNormalization()(x)
        x = merge([x, conv], 'sum')

        final = Convolution2D(6, 1, 1, activation='linear', border_mode='same', init='he_normal')(x)

        self.model = Model(input=inputs, output=final)

        print("Loading weights...")
        self.model.load_weights("{0}_weights.hdf5".format(self.root))

    def validation_generator(self):        

        self.median_i = np.median(self.images[:,100:-100,100:-100])

        input_validation = np.zeros((self.batch_size,self.nx,self.ny,2), dtype='float32')

        while 1:
            for i in range(self.n_frames):

                print('{0}/{1}'.format(i,self.n_frames))

                input_validation[:,:,:,0] = self.images[i*self.batch_size:(i+1)*self.batch_size,:,:] / self.median_i
                input_validation[:,:,:,1] = self.images[i*self.batch_size+1:(i+1)*self.batch_size+1,:,:] / self.median_i                

                yield input_validation        

    def predict_validation(self):
        print("Predicting validation data...")

        tmp = np.load('/net/duna/scratch1/aasensio/deepLearning/opticalFlow/database/normalization.npz')
        min_i, max_i, min_v, max_v = tmp['arr_0'], tmp['arr_1'], tmp['arr_2'], tmp['arr_3']

        # ff = io.readsav(self.observations)
        # im = ff['cont']

        # x = np.arange(self.nx)
        # y = np.arange(self.ny)

        start = time.time()
        out = self.model.predict_generator(self.validation_generator(), self.n_frames, max_q_size=1)
        end = time.time()

        print("Prediction took {0} seconds...".format(end-start))

        ff = h5py.File(self.output, 'w')
        db = ff.create_dataset("velocity", (self.n_frames, self.nx, self.ny, 6), dtype='float32')

        for i in range(6):
            out[:,:,:,i] = out[:,:,:,i] * (max_v[i] - min_v[i]) + min_v[i]

        db[:] = out[:]

        db = ff.create_dataset("vx", (3, self.nx, self.ny), dtype='float32')
        db[:] = self.vx

        db = ff.create_dataset("vy", (3, self.nx, self.ny), dtype='float32')
        db[:] = self.vy
        ff.close()

    def compare_velocities(self):

        ff = h5py.File(self.output)
        prediction = ff.get("velocity")

        f, ax = pl.subplots(nrows=2, ncols=2, figsize=(10,10))

        im = ax[0,0].imshow(self.vx[0,200:300,200:300] / 1e5)
        pl.colorbar(im, ax=ax[0,0])

        im = ax[0,1].imshow(10*prediction[0,200:300,200:300,0])
        pl.colorbar(im, ax=ax[0,1])

        im = ax[1,0].imshow(self.vy[0,200:300,200:300] / 1e5)
        pl.colorbar(im, ax=ax[1,0])

        im = ax[1,1].imshow(10*prediction[0,200:300,200:300,1])
        pl.colorbar(im, ax=ax[1,1])

        pl.show()


        stop()

if (__name__ == '__main__'):
    
    # out = trainDNNFull('../training/cnns/resnet', 'cont.idl', 'imax_velocity.h5', 'cont')
    out = trainDNNFull('../../training/cnns/resnet2', 'mancha_velocity.h5', 'mov')
    out.defineNetwork()
    out.predict_validation()
    #out.compare_velocities()
