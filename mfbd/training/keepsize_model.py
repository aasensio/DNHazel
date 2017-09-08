from keras.layers import Input, Conv2D, Activation, BatchNormalization, GaussianNoise, add, UpSampling2D
from keras.models import Model

def residual(inputs, n_filters):
    x = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = add([x, inputs])

    return x    

def residual_v2(inputs, n_filters):
    x = BatchNormalization()(inputs)
    x = Activation('relu')(x)
    x = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)    
    x = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    
    x = add([x, inputs])

    return x    

def define_network(nx, ny, noise, depth):
    inputs = Input(shape=(nx, ny, 12))
    x = GaussianNoise(noise)(inputs)

    x = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    for i in range(depth):
        x = residual(x, 64)

    x = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    final = Conv2D(1, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal')(x)

    return Model(inputs=inputs, outputs=final)