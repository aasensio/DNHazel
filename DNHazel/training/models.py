from keras.layers import Input, Dense, Conv1D, BatchNormalization, Activation, add, concatenate, Flatten, GaussianNoise, GlobalAveragePooling1D
from keras.models import Model
from keras.regularizers import l2
from keras.layers.advanced_activations import ELU
from concrete_dropout import ConcreteDropout

def residual(inputs, n_filters, activation, l2_reg, strides):
    x0 = Conv1D(n_filters, 1, padding='same', kernel_initializer='he_normal', strides=strides, kernel_regularizer=l2(l2_reg))(inputs)

    x = Conv1D(n_filters, 3, padding='same', kernel_initializer='he_normal', strides=strides, kernel_regularizer=l2(l2_reg))(inputs)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)    
    x = Conv1D(n_filters, 3, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = add([x0, x])

    return x

def residual_dropout(inputs, n_filters, activation, l2_reg, strides, wd, dd):    
    x0 = ConcreteDropout(Conv1D(n_filters, 1, padding='same', kernel_initializer='he_normal', strides=strides, kernel_regularizer=l2(l2_reg)), 
        weight_regularizer=wd, dropout_regularizer=dd)(inputs)
    
    x = ConcreteDropout(Conv1D(n_filters, 3, padding='same', kernel_initializer='he_normal', strides=strides, kernel_regularizer=l2(l2_reg)), 
        weight_regularizer=wd, dropout_regularizer=dd)(inputs)

    x = BatchNormalization()(x)
    x = Activation(activation)(x)    
    x = ConcreteDropout(Conv1D(n_filters, 3, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg)),
        weight_regularizer=wd, dropout_regularizer=dd)(x)

    x = BatchNormalization()(x)
    x = add([x0, x])

    return x

def elu_modif(x, a=1.):
    e = 1e-15
    return ELU(alpha=a)(x)+1.+e

def network(n_lambda, depth, noise=0.0, activation='relu', n_filters=64, l2_reg=1e-7, n_mixture=8):
      
    stokes_input = Input(shape=(n_lambda,4), name='stokes_input')
    x = GaussianNoise(noise)(stokes_input)

    x = Conv1D(n_filters, 3, activation=activation, padding='same', kernel_initializer='he_normal', name='conv_1', kernel_regularizer=l2(l2_reg))(x)

    for i in range(depth):
        x = residual(x, n_filters*(i+1), activation, l2_reg, strides=2)
    
    intermediate = Flatten(name='flat')(x)

    mu_input = Input(shape=(1,), name='mu_input')

    intermediate_conv = concatenate([intermediate, mu_input], name='FC')

    out_mu = Dense(n_mixture*9, name='FC_mu')(intermediate_conv)
    out_sigma = Dense(n_mixture*9, activation=elu_modif, name='FC_sigma')(intermediate_conv)
    out_alpha = Dense(n_mixture, activation='softmax', name='FC_alpha')(intermediate_conv)

    out = concatenate([out_mu, out_sigma, out_alpha])

    model = Model(inputs=[stokes_input, mu_input], outputs=out)

    return model

def network_dropout(n_lambda, n_classes, depth, noise=0.0, activation='relu', n_filters=64, l2_reg=1e-7, wd=0.0, dd=0.0):
      
    stokes_input = Input(shape=(n_lambda,4), name='stokes_input')
    x = GaussianNoise(noise)(stokes_input)

    x = ConcreteDropout(Conv1D(n_filters, 3, activation=activation, padding='same', kernel_initializer='he_normal', name='conv_1', 
        kernel_regularizer=l2(l2_reg)),weight_regularizer=wd, dropout_regularizer=dd)(x)

    for i in range(depth):
        x = residual_dropout(x, n_filters*(i+1), activation, l2_reg, 2, wd, dd)
    
    intermediate = Flatten(name='flat')(x)

    mu_input = Input(shape=(1,), name='mu_input')

    intermediate_conv = concatenate([intermediate, mu_input], name='FC')

    x = ConcreteDropout(Dense(3*n_classes, activation='relu', name='FC_tau'),weight_regularizer=wd, dropout_regularizer=dd)(intermediate_conv)

    out_tau = ConcreteDropout(Dense(n_classes, activation='softmax', name='out_tau'),weight_regularizer=wd, dropout_regularizer=dd)(x)
    out_v = ConcreteDropout(Dense(n_classes, activation='softmax', name='out_v'),weight_regularizer=wd, dropout_regularizer=dd)(x)
    out_vth = ConcreteDropout(Dense(n_classes, activation='softmax', name='out_vth'),weight_regularizer=wd, dropout_regularizer=dd)(x)
    out_a = ConcreteDropout(Dense(n_classes, activation='softmax', name='out_a'),weight_regularizer=wd, dropout_regularizer=dd)(x)

    out_B = ConcreteDropout(Dense(n_classes, activation='softmax', name='out_B'),weight_regularizer=wd, dropout_regularizer=dd)(x)
    out_thB = ConcreteDropout(Dense(n_classes, activation='softmax', name='out_thB'),weight_regularizer=wd, dropout_regularizer=dd)(x)
    out_phiB = ConcreteDropout(Dense(n_classes, activation='softmax', name='out_phiB'),weight_regularizer=wd, dropout_regularizer=dd)(x)
    out_ThB = ConcreteDropout(Dense(n_classes, activation='softmax', name='out_ThB'),weight_regularizer=wd, dropout_regularizer=dd)(x)
    out_PhiB = ConcreteDropout(Dense(n_classes, activation='softmax', name='out_PhiB'),weight_regularizer=wd, dropout_regularizer=dd)(x)

    model = Model(inputs=[stokes_input, mu_input], outputs=[out_tau, out_v, out_vth, out_a, out_B, out_thB, out_phiB, out_ThB, out_PhiB])

    return model

def network_nodropout(n_lambda, n_classes, depth, noise=0.0, activation='relu', n_filters=64, l2_reg=1e-7):
      
    stokes_input = Input(shape=(n_lambda,1), name='stokes_input')
    x = GaussianNoise(noise)(stokes_input)

    x = Conv1D(n_filters, 3, activation=activation, padding='same', kernel_initializer='he_normal', name='conv_1', 
        kernel_regularizer=l2(l2_reg))(x)

    for i in range(depth):
        x = residual(x, n_filters*(i+1), activation, l2_reg, 1)
    
    intermediate = GlobalAveragePooling1D(name='flat')(x)

    mu_input = Input(shape=(1,), name='mu_input')

    x = concatenate([intermediate, mu_input], name='FC')

    out_tau = Dense(n_classes, activation='softmax', name='out_tau')(x)
    out_v = Dense(n_classes, activation='softmax', name='out_v')(x)
    out_vth = Dense(n_classes, activation='softmax', name='out_vth')(x)
    out_a = Dense(n_classes, activation='softmax', name='out_a')(x)

    # out_B = Dense(n_classes, activation='softmax', name='out_B')(x)
    # out_thB = Dense(n_classes, activation='softmax', name='out_thB')(x)
    # out_phiB = Dense(n_classes, activation='softmax', name='out_phiB')(x)
    # out_ThB = Dense(n_classes, activation='softmax', name='out_ThB')(x)
    # out_PhiB = Dense(n_classes, activation='softmax', name='out_PhiB')(x)

    # model = Model(inputs=[stokes_input, mu_input], outputs=[out_tau, out_v, out_vth, out_a, out_B, out_thB, out_phiB, out_ThB, out_PhiB])
    model = Model(inputs=[stokes_input, mu_input], outputs=[out_tau, out_v, out_vth, out_a])

    return model
