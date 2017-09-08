import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
import matplotlib.pyplot as pl
import keras.backend as K
from keras.callbacks import CSVLogger
from keras.layers import Input, Lambda, Dense, Flatten, BatchNormalization, Activation, Conv1D, add, concatenate
from keras.models import Model
from keras.optimizers import Adam
from sklearn.cluster import KMeans, AgglomerativeClustering
import pandas as pd

def residual(inputs, n_filters, activation, strides):
    x0 = Conv1D(n_filters, 1, padding='same', kernel_initializer='he_normal', strides=strides)(inputs)

    x = Conv1D(n_filters, 3, padding='same', kernel_initializer='he_normal', strides=strides)(inputs)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)    
    x = Conv1D(n_filters, 3, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = add([x0, x])

    return x


def sample_center_points(y, method='all', k=100, keep_edges=False):
    """
    function to define kernel centers with various downsampling alternatives
    """

    # make sure y is 1D
    y = y.ravel()

    # keep all points as kernel centers
    if method is 'all':
        return y

    # retain outer points to ensure expressiveness at the target borders
    if keep_edges:
        y = np.sort(y)
        centers = np.array([y[0], y[-1]])
        y = y[1:-1]
        # adjust k such that the final output has size k
        k -= 2
    else:
        centers = np.empty(0)

    if method is 'random':
        cluster_centers = np.random.choice(y, k, replace=False)

    # iteratively remove part of pairs that are closest together until everything is at least 'd' apart
    elif method is 'distance':
        raise NotImplementedError

    # use 1-D k-means clustering
    elif method is 'k_means':
        model = KMeans(n_clusters=k, n_jobs=-2)
        model.fit(y.reshape(-1, 1))
        cluster_centers = model.cluster_centers_

    # use agglomerative clustering
    elif method is 'agglomerative':
        model = AgglomerativeClustering(n_clusters=k, linkage='complete')
        model.fit(y.reshape(-1, 1))
        labels = pd.Series(model.labels_, name='label')
        y_s = pd.Series(y, name='y')
        df = pd.concat([y_s, labels], axis=1)
        cluster_centers = df.groupby('label')['y'].mean().values

    else:
        raise ValueError("unknown method '{}'".format(method))

    return np.append(centers, cluster_centers)


class kmn_many_sigma(object):
    def __init__(self, x_train, y_train, center_sampling_method='k_means', n_centers=20, sigmas=None, keep_edges=True, 
                 estimator=None, validation_set=None, batch_size=32):
        self.center_sampling_method = center_sampling_method
        self.n_centers = n_centers
        self.batch_size = batch_size
        if (sigmas is None):
            self.sigmas = np.ones(self.n_centers).astype('float32')
        else:
            self.n_sigma = len(sigmas)
            self.sigmas = np.repeat(sigmas, self.n_centers).astype('float32')
        
        self.keep_edges = keep_edges
        self.center_locs = sample_center_points(y_train, method=self.center_sampling_method, 
                                                k=self.n_centers, keep_edges=self.keep_edges).astype('float32')
        
        self.n_data, self.n_features = x_train.shape
        
        self.train = []
        self.train.append(x_train.reshape(self.n_data, self.n_features).astype('float32'))
        self.train.append(y_train.reshape(self.n_data, 1).astype('float32'))
        
        self.validation_present = False
        if (validation_set != None):
            self.validation_present = True
            x_val = validation_set['x']
            y_val = validation_set['y']
            self.n_data_val, _ = x_val.shape
            self.validation = []
            self.validation.append(x_val.reshape(self.n_data_val, self.n_features).astype('float32'))
            self.validation.append(y_val.reshape(self.n_data_val, 1).astype('float32'))
            
        self.oneDivSqrtTwoPI = 1.0 / np.sqrt(2.0*np.pi) # normalisation factor for gaussian.
            
    def gaussian_distribution(self, y, mu, sigma):
        result = (y - mu) / sigma
        result = - 0.5 * (result * result)
        return (K.exp(result) / sigma) * self.oneDivSqrtTwoPI
    
    def mdn_loss_function(self, args):
        y, weights = args
        result = self.gaussian_distribution(y, self.center_locs, self.sigmas) * weights
        result = K.sum(result, axis=1)
        result = - K.log(result)
        return K.mean(result)
    
    def estimator_many_sigma(self, depth, n_filters, n_center, n_sigma):

# Inputs
        input_x = Input(shape=(self.n_lambda,4), name='stokes_input')
        y_true = Input(shape=(1,), name='y_true')
        mu_input = Input(shape=(1,), name='mu_input')

# Neural network
        x = Conv1D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv_1')(input_x)

        for i in range(depth):
            x = residual(x, n_filters*(i+1), 'relu', strides=2)
    
        intermediate = Flatten(name='flat')(x)
        intermediate_conv = concatenate([intermediate, mu_input], name='FC')

# Output weights
        weights = Dense(self.n_centers*self.n_sigma, activation='softmax', name='weights')(intermediate_conv)

# Definition of the loss function
        loss = Lambda(self.mdn_loss_function, output_shape=(1,), name='loss')([y_true, weights])
        
        self.model = Model(inputs=[input_x, y_true], outputs=[loss])
        self.model.add_loss(loss)
    
# Compile with the loss weight set to None, so it will be omitted
        self.model.compile(loss=[None], loss_weights=[None], optimizer=Adam(lr=0.01))
        self.model.summary()

# Now generate a second network that ends up in the weights for later evaluation        
        layer_name = 'weights'
        self.output_weights = Model(inputs=self.model.input,
                                 outputs=self.model.get_layer(layer_name).output)
                       
    def fit(self):
        self.estimator_many_sigma(self.n_centers, self.n_sigma)
        cb = CSVLogger("training.csv")
        self.model.fit(x=self.train, epochs=300, batch_size=3750, callbacks=[cb], 
                       validation_data=(self.validation, None))
        
                
    def predict_density(self, x_test):
        y = np.linspace(-10,10,300)
        weights = self.model.predict(x_test)
        result = self.gaussian_distribution(torch.unsqueeze(y,1), self.center_locs, self.sigma) * weights
        result = torch.sum(result, dim=1)
        return y.data.numpy(), result
    
    def sample_density(self, x_test):
        test = []
        test.append(x_test)
        test.append(x_test)
        
        weights = self.output_weights.predict(test)
        print(weights.shape)
        locs = self.center_locs
        sigma = self.sigmas
        n = len(x_test)
        out = np.zeros(n)
        for i in range(n):
            ind = np.random.choice(self.n_centers * self.n_sigma, p=weights[i,:])
            out[i] = np.random.normal(loc=locs[ind], scale=sigma[ind])
        return out
    
    def plot_loss(self):
        out = pd.read_csv('training.csv').as_matrix()
        f, ax = pl.subplots()
        ax.plot(out[:,1], label='Training set')
        if (self.validation_present):
            ax.plot(out[:,2], label='Validation set')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.legend()