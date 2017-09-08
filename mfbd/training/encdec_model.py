from keras.layers import Input, Conv2D, Activation, BatchNormalization, GaussianNoise, add, UpSampling2D
from keras.models import Model

def residual(inputs, n_filters):
    x = BatchNormalization()(inputs)
    x = Activation('relu')(x)
    x = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)    
    x = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer='he_normal')(x)    
    x = add([x, inputs])

    return x    

def residual_down(inputs, n_filters):
    x = BatchNormalization()(inputs)
    x = Activation('relu')(x)
    x = Conv2D(n_filters, (3, 3), strides=2, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)    
    x = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer='he_normal')(x)

    shortcut = Conv2D(n_filters, (1, 1), strides=2, padding='same', kernel_initializer='he_normal')(inputs)

    x = add([x, shortcut])

    return x

def residual_up(inputs, n_filters):
    x_up = UpSampling2D(size=(2,2))(inputs)

    x = BatchNormalization()(x_up)
    x = Activation('relu')(x)
    x = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)    
    x = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer='he_normal')(x)

    shortcut = Conv2D(n_filters, (1, 1), padding='same', kernel_initializer='he_normal')(x_up)

    x = add([x, shortcut])

    return x

def define_network(nx, ny, noise, depth, n_filters):

    inputs = Input(shape=(nx, ny, 12))

# (88,88,12)

    x = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

# (88,88,64)

    x = residual_down(x, n_filters*2)   # (44,44,128)
    x = residual_down(x, n_filters*4)   # (22,22,256)
    x = residual_down(x, n_filters*4)   # (11,11,512)

    for i in range(depth):
        x = residual(x, n_filters*4)

    x = residual_up(x, n_filters*4)   # (22,22,256)
    x = residual_up(x, n_filters*2)   # (44,44,128)
    x = residual_up(x, n_filters)     # (88,88,64)

    x = residual(x, n_filters)

    final = Conv2D(1, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal')(x)

    return Model(inputs=inputs, outputs=final)