from keras.layers import Input, Conv1D, BatchNormalization, Dense, LSTM, TimeDistributed, Flatten, Bidirectional
from keras.models import Model
from keras.regularizers import l2

def zdi(n_lambda, noise, activation='relu', n_filters=64, l2_reg=1e-7):
    """
    Deep residual recurrent network for the inversion of stellar spectra
    """
    st = Input(shape=(None, n_lambda, 1), name='stokes_input')
    x = BatchNormalization()(st)

#    x = TimeDistributed(Conv1D(n_filters, 3, activation=activation, padding='same', strides=1, kernel_initializer='he_normal', 
#           kernel_regularizer=l2(l2_reg)), name='conv_1')(x)
#    x = BatchNormalization()(x)
    
#    x = TimeDistributed(Conv1D(2*n_filters, 3, activation=activation, padding='same', strides=2, kernel_initializer='he_normal', 
#          kernel_regularizer=l2(l2_reg)), name='conv_2')(x)
#    x = BatchNormalization()(x)

#    x = TimeDistributed(Conv1D(4*n_filters, 3, activation=activation, padding='same', strides=2, kernel_initializer='he_normal', 
#         kernel_regularizer=l2(l2_reg)), name='conv_3')(x)
#    x = BatchNormalization()(x)
    
    x = TimeDistributed(Flatten(), name='flatten')(x)

    # x_modulus = LSTM(64)(x)
    # output_modulus = Dense(1, name='modulus')(x_modulus)

    # x_alpha = LSTM(128, return_sequences=True)(x)
    x = LSTM(128)(x)
    # x = Dense(128, kernel_initializer='he_normal')(x)
    output_alpha = Dense(5, name='alpha')(x)

    # x_beta = LSTM(128)(x)
    # output_beta = Dense(121, name='beta')(x_beta)

    # x_gamma = LSTM(128)(x)
    # output_gamma = Dense(121, name='gamma')(x_gamma)

    # return Model(inputs=stokes_input, outputs=[output_modulus, output_alpha, output_beta, output_gamma])
    return Model(inputs=st, outputs=output_alpha)
