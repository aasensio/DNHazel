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


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / N 

class trainDNNFull(object):

    def __init__(self, root, observations, output, name_of_variable):

# Only allocate needed memory
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        session = tf.Session(config=config)
        ktf.set_session(session)

        self.root = root
        self.nx = 800
        self.ny = 800
        self.n_times = 2
        self.n_filters = 64
        self.batch_size = 1
        self.n_conv_layers = 20
        self.stride = 1
        self.skip_frequency = 2
        self.n_frames = 57
        self.observations = observations
        self.output = output
        self.name_of_variable = name_of_variable
        
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
        f = io.readsav(self.observations)
        out = f[self.name_of_variable]

        self.median_i = np.median(out[:,100:-100,100:-100])

        input_validation = np.zeros((self.batch_size,self.nx,self.ny,2), dtype='float32')

        while 1:
            for i in range(self.n_frames):

                print('{0}/{1}'.format(i,self.n_frames))

                input_validation[:,:,:,0] = out[i*self.batch_size:(i+1)*self.batch_size,100:100+self.nx,100:100+self.ny] / self.median_i
                input_validation[:,:,:,1] = out[i*self.batch_size+1:(i+1)*self.batch_size+1,100:100+self.nx,100:100+self.ny] / self.median_i

                yield input_validation

        f.close()

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
        ff.close()

        # print("Plotting validation data...")

        # f, ax = pl.subplots(nrows=1, ncols=3, figsize=(15,7))

        # label = ['1', '0.1', '0.01']
        
        # FFMpegWriter = manimation.writers['ffmpeg']
        # metadata = dict(title='Movie Test', artist='Matplotlib', comment='Movie support!')
        # writer = FFMpegWriter(fps=3, bitrate=5000, metadata=metadata)
        # with writer.saving(f, "movie.mp4", self.n_frames):
        #     for i in range(self.n_frames):
        #         print ("Frame {0}/{1}".format(i,self.n_frames))
        #         for j in range(3):
        #             ax[j].imshow(im[i,100:100+self.nx,100:100+self.ny], cmap=pl.cm.gray)
        #             ax[j].streamplot(x,y,out[i,:,:,2*j], out[i,:,:,2*j+1],density=1)        
        #             ax[j].set_xlim([0,self.nx])
        #             ax[j].set_ylim([0,self.ny])
        #             ax[j].set_title(r'$\tau=${0}'.format(label[j]))
        
        #         writer.grab_frame()
        #         for j in range(3):
        #             ax[j].cla()

        stop()
        
        # pl.close('all')   

        # f, ax = pl.subplots(nrows=4, ncols=3, figsize=(18,12))

        # tau_label = ['1', '0.1', '0.01']
                
        # for i in range(4):
        #     for j in range(3):
        #         res = ax[i,j].imshow(im[i*5,100:100+self.nx,100:100+self.ny], cmap=pl.cm.gray)
        #         ax[i,j].streamplot(x,y,out[i,:,:,2*j]*(max_v[2*j]-min_v[2*j])+min_v[2*j], out[i,:,:,2*j+1]*(max_v[2*j+1]-min_v[2*j+1])+min_v[2*j+1],density=1)
        #         # ax[i,j].quiver(np.arange(20)*5, np.arange(20)*5, 10*out[i,:,:,2*j]*(max_v[2*j]-min_v[2*j])+min_v[2*j], 10*out[i,:,:,2*j+1]*(max_v[2*j+1]-min_v[2*j+1])+min_v[2*j+1])
        #         pl.colorbar(res, ax=ax[i,j])
        #         ax[i,j].set_xlim([0,self.nx])
        #         ax[i,j].set_ylim([0,self.ny])
        #         ax[i,j].set_title(r'$\tau=${0}'.format(tau_label[j]))

        # pl.tight_layout()

        # pl.show()        

        # # pl.savefig("{0}_imax.png".format(self.root))

        # stop()
            
if (__name__ == '__main__'):
    
    # out = trainDNNFull('../training/cnns/resnet', 'cont.idl', 'imax_velocity.h5', 'cont')
    out = trainDNNFull('../training/cnns/resnet2', '/net/vena/scratch1/deepLearning/opticalFlow/database/sf_Icon_307-364.sav', 'imax_velocity_noPmodes.h5', 'mov')
    out.defineNetwork()
    out.predict_validation()
