import numpy as np
import matplotlib.pyplot as pl
import platform
import os
import json
from ipdb import set_trace as stop
import cv2

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from keras.layers import Input, Conv2D, Dropout, Flatten, Dense, BatchNormalization, Activation, UpSampling2D, Reshape
from keras.callbacks import ModelCheckpoint, Callback, ReduceLROnPlateau
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, model_from_json
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.datasets import mnist
import tensorflow as tf
import keras.backend.tensorflow_backend as K
import time


class gan(object):

    def __init__(self):

# Only allocate needed memory
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        session = tf.Session(config=config)
        K.set_session(session)

        self.dropout = 0.4
        self.depth = 64
        self.batch_size = 32
        self.zed = 100

        (x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data(path='/net/vena/scratch/Dropbox/GIT/DeepLearning/GAN/mnist.dat')

        self.n_train, self.nx, self.ny = x_train.shape
        self.n_test, self.nx, self.ny = self.x_test.shape

        self.x_train = np.zeros((self.n_train, self.nx, self.ny, 1), dtype='float32')
        self.x_train[:,:,:,0] = x_train

    def define_networks(self):
        inputs = Input(shape=(self.nx, self.ny, 1))
        x = Conv2D(self.depth, (3,3), strides=2, padding='same', kernel_initializer='he_normal', activation=LeakyReLU(alpha=0.2))(inputs)
        x = Dropout(self.dropout)(x)
        x = Conv2D(self.depth*2, (3,3), strides=2, padding='same', kernel_initializer='he_normal', activation=LeakyReLU(alpha=0.2))(x)
        x = Dropout(self.dropout)(x)
        x = Conv2D(self.depth*4, (3,3), strides=2, padding='same', kernel_initializer='he_normal', activation=LeakyReLU(alpha=0.2))(x)
        x = Dropout(self.dropout)(x)
        x = Conv2D(self.depth*8, (3,3), strides=1, padding='same', kernel_initializer='he_normal', activation=LeakyReLU(alpha=0.2))(x)
        x = Dropout(self.dropout)(x)
        x = Flatten()(x)
        x = Dense(1, activation='sigmoid')(x)

        self.discriminator = Model(inputs=inputs, outputs=x)

        dim = 7
        depth = 64 * 4
        inputs = Input(shape=(100,))
        x = Dense(dim * dim * depth)(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Reshape((dim, dim, depth))(x)
        x = Dropout(self.dropout)(x)
        
        x = UpSampling2D(size=(2,2))(x)        
        x = Conv2D(int(depth / 2) , (3,3), padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Activation('relu')(x)

        x = UpSampling2D(size=(2,2))(x)        
        x = Conv2D(int(depth / 4) , (3,3), padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Activation('relu')(x)

        x = Conv2D(int(depth / 8) , (3,3), padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Activation('relu')(x)

        x = Conv2D(1 , (3,3), padding='same', kernel_initializer='he_normal')(x)
        x = Activation('sigmoid')(x)

        self.generator = Model(inputs=inputs, outputs=x)

        noise = Input(shape=self.generator.input_shape[1:])

    def gan(self):
    # initialize a GAN trainer

    # this is the fastest way to train a GAN in Keras
    # two models are updated simutaneously in one pass

        noise = Input(shape=self.generator.input_shape[1:])
        real_data = Input(shape=self.discriminator.input_shape[1:])

        generated = self.generator(noise)
        gscore = self.discriminator(generated)
        rscore = self.discriminator(real_data)

        def log_eps(i):
            return K.log(i+1e-11)

        # single side label smoothing: replace 1.0 with 0.9
        dloss = - K.mean(log_eps(1-gscore) + .1 * log_eps(1-rscore) + .9 * log_eps(rscore))
        gloss = - K.mean(log_eps(gscore))

        Adam = tf.train.AdamOptimizer

        lr,b1 = 1e-4,.2 # otherwise won't converge.
        optimizer = Adam(lr)

        grad_loss_wd = optimizer.compute_gradients(dloss, self.discriminator.trainable_weights)
        update_wd = optimizer.apply_gradients(grad_loss_wd)

        grad_loss_wg = optimizer.compute_gradients(gloss, self.generator.trainable_weights)
        update_wg = optimizer.apply_gradients(grad_loss_wg)

        def get_internal_updates(model):
            # get all internal update ops (like moving averages) of a model
            inbound_nodes = model.inbound_nodes
            input_tensors = []
            for ibn in inbound_nodes:
                input_tensors+= ibn.input_tensors
            updates = [model.get_updates_for(i) for i in input_tensors]
            return updates

        other_parameter_updates = [get_internal_updates(m) for m in [self.discriminator,self.generator]]
        # those updates includes batch norm.

        print('other_parameter_updates for the models(mainly for batch norm):')
        print(other_parameter_updates)

        train_step = [update_wd, update_wg, other_parameter_updates]
        losses = [dloss,gloss]

        learning_phase = K.learning_phase()

        def gan_feed(sess,batch_image,z_input):
            # actual GAN trainer
            nonlocal train_step,losses,noise,real_data,learning_phase

            res = sess.run([train_step,losses],feed_dict={
            noise:z_input,
            real_data:batch_image,
            learning_phase:True,
            # Keras layers needs to know whether
            # this run is training or testring (you know, batch norm and dropout)
            })

            loss_values = res[1]
            return loss_values #[dloss,gloss]

        return gan_feed


    def train(self, ep=2000,noise_level=.01):

        gan_feed = self.gan()
        sess = K.get_session()

        np.random.shuffle(self.x_train)
        shuffled_cifar = self.x_train
        length = len(shuffled_cifar)

        for i in range(ep):
            j = i % int(length / self.batch_size)
            minibatch = shuffled_cifar[j*self.batch_size:(j+1)*self.batch_size]

            z_input = np.random.normal(loc=0.,scale=1.,size=(self.batch_size,self.zed))

            # train for one step
            losses = gan_feed(sess,minibatch,z_input)
            if (i % 10 == 0):
                print('{0} - dloss:{1:6.4f} gloss:{2:6.4f}'.format(i,losses[0],losses[1]))

            
        self.show()
    
    def show(self, save=False):
        i = np.random.normal(loc=0.,scale=1.,size=(self.batch_size,self.zed))
        gened = self.generator.predict([i])
        stop()


if (__name__ == '__main__'):
    
    out = gan()
    out.define_networks()
    out.train()