from keras.layers import Input, Conv2D, Activation, BatchNormalization, GaussianNoise, add, UpSampling2D, concatenate, Conv2DTranspose, Lambda
from keras.models import Model
from keras.regularizers import l2
import tensorflow as tf
from keras.engine.topology import Layer
from keras.engine import InputSpec
from keras.utils import conv_utils
import keras.backend as K

def spatial_reflection_2d_padding(x, padding=((1, 1), (1, 1)), data_format=None):
    """Pads the 2nd and 3rd dimensions of a 4D tensor.
    # Arguments
        x: Tensor or variable.
        padding: Tuple of 2 tuples, padding pattern.
        data_format: One of `channels_last` or `channels_first`.
    # Returns
        A padded 4D tensor.
    # Raises
        ValueError: if `data_format` is neither
            `channels_last` or `channels_first`.
    """
    assert len(padding) == 2
    assert len(padding[0]) == 2
    assert len(padding[1]) == 2
    if data_format is None:
        data_format = image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format ' + str(data_format))

    if data_format == 'channels_first':
        pattern = [[0, 0],
                   [0, 0],
                   list(padding[0]),
                   list(padding[1])]
    else:
        pattern = [[0, 0],
                   list(padding[0]), list(padding[1]),
                   [0, 0]]
    return tf.pad(x, pattern, "REFLECT")

class ReflectionPadding2D(Layer):
    """Reflection-padding layer for 2D input (e.g. picture).
    This layer can add rows and columns or zeros
    at the top, bottom, left and right side of an image tensor.
    # Arguments
        padding: int, or tuple of 2 ints, or tuple of 2 tuples of 2 ints.
            - If int: the same symmetric padding
                is applied to width and height.
            - If tuple of 2 ints:
                interpreted as two different
                symmetric padding values for height and width:
                `(symmetric_height_pad, symmetric_width_pad)`.
            - If tuple of 2 tuples of 2 ints:
                interpreted as
                `((top_pad, bottom_pad), (left_pad, right_pad))`
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
    # Input shape
        4D tensor with shape:
        - If `data_format` is `"channels_last"`:
            `(batch, rows, cols, channels)`
        - If `data_format` is `"channels_first"`:
            `(batch, channels, rows, cols)`
    # Output shape
        4D tensor with shape:
        - If `data_format` is `"channels_last"`:
            `(batch, padded_rows, padded_cols, channels)`
        - If `data_format` is `"channels_first"`:
            `(batch, channels, padded_rows, padded_cols)`
    """
    
    def __init__(self,
                 padding=(1, 1),
                 data_format=None,
                 **kwargs):
        super(ReflectionPadding2D, self).__init__(**kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)
        if isinstance(padding, int):
            self.padding = ((padding, padding), (padding, padding))
        elif hasattr(padding, '__len__'):
            if len(padding) != 2:
                raise ValueError('`padding` should have two elements. '
                                 'Found: ' + str(padding))
            height_padding = conv_utils.normalize_tuple(padding[0], 2,
                                                        '1st entry of padding')
            width_padding = conv_utils.normalize_tuple(padding[1], 2,
                                                       '2nd entry of padding')
            self.padding = (height_padding, width_padding)
        else:
            raise ValueError('`padding` should be either an int, '
                             'a tuple of 2 ints '
                             '(symmetric_height_pad, symmetric_width_pad), '
                             'or a tuple of 2 tuples of 2 ints '
                             '((top_pad, bottom_pad), (left_pad, right_pad)). '
                             'Found: ' + str(padding))
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            if input_shape[2] is not None:
                rows = input_shape[2] + self.padding[0][0] + self.padding[0][1]
            else:
                rows = None
            if input_shape[3] is not None:
                cols = input_shape[3] + self.padding[1][0] + self.padding[1][1]
            else:
                cols = None
            return (input_shape[0],
                    input_shape[1],
                    rows,
                    cols)
        elif self.data_format == 'channels_last':
            if input_shape[1] is not None:
                rows = input_shape[1] + self.padding[0][0] + self.padding[0][1]
            else:
                rows = None
            if input_shape[2] is not None:
                cols = input_shape[2] + self.padding[1][0] + self.padding[1][1]
            else:
                cols = None
            return (input_shape[0],
                    rows,
                    cols,
                    input_shape[3])

    def call(self, inputs):
        return spatial_reflection_2d_padding(inputs,
                                    padding=self.padding,
                                    data_format=self.data_format)

    def get_config(self):
        config = {'padding': self.padding,
                  'data_format': self.data_format}
        base_config = super(ReflectionPadding2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# def padding(block_function, filters, repetitions, is_first_layer=False):
    """Builds a residual block with repeating bottleneck blocks.
    """
    # def f(input):

def convolution(n_filters, kernel_size=3, l2_reg=0.0, strides=1):
    def fun(inputs):
        x = BatchNormalization()(inputs)
        x = Activation('relu')(x)
        # x = ReflectionPadding2D(strides)(x)
        x = Conv2D(n_filters, (kernel_size, kernel_size), padding='same', kernel_initializer='he_normal', 
            kernel_regularizer=l2(l2_reg), strides=strides)(x)
        return x
    return fun

def transposed_convolution(n_filters, kernel_size=3, l2_reg=0.0, strides=1):
    def fun(inputs):
        x = BatchNormalization()(inputs)
        x = Activation('relu')(x)
        # x = ReflectionPadding2D(strides)(x)
        x = Conv2DTranspose(n_filters, (kernel_size, kernel_size), padding='same', kernel_initializer='he_normal', 
            kernel_regularizer=l2(l2_reg), strides=strides)(x)
        return x
    return fun

    
def deconvolution_module(nx, ny, l2_reg):
    def fun(estimation, next_frame, t1, t2, t3):

        inputs = concatenate([estimation, next_frame])
    
        A01 = convolution(64, kernel_size=3, l2_reg=l2_reg, strides=1)(inputs)

        C11 = convolution(64, kernel_size=3, l2_reg=l2_reg, strides=2)(A01)
        C12 = convolution(64, kernel_size=3, l2_reg=l2_reg, strides=1)(C11)
        C13 = convolution(64, kernel_size=3, l2_reg=l2_reg, strides=1)(C12)
        C14 = convolution(64, kernel_size=3, l2_reg=l2_reg, strides=1)(C13)
        C14 = add([C11, C14])

        C21 = convolution(64, kernel_size=3, l2_reg=l2_reg, strides=1)(C14)
        C22 = convolution(64, kernel_size=3, l2_reg=l2_reg, strides=1)(C21)
        C23 = convolution(64, kernel_size=3, l2_reg=l2_reg, strides=1)(C22)
        C24 = convolution(64, kernel_size=3, l2_reg=l2_reg, strides=1)(C23)
        C24 = add([C21, C24])

        C31 = convolution(128, kernel_size=3, l2_reg=l2_reg, strides=2)(C24)
        C32 = convolution(128, kernel_size=3, l2_reg=l2_reg, strides=1)(C31)
        C33 = convolution(128, kernel_size=3, l2_reg=l2_reg, strides=1)(C32)
        C34 = convolution(128, kernel_size=3, l2_reg=l2_reg, strides=1)(C33)
        C34 = add([C31, C34])

        C41 = convolution(256, kernel_size=3, l2_reg=l2_reg, strides=2)(C34)
        C42 = Lambda(lambda x: K.concatenate([x,t1],axis=-1), output_shape=(nx/8,ny/8,512))(C41)
        B42 = Conv2D(256, (1, 1), padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), strides=1)(C42)
        C43 = convolution(256, kernel_size=3, l2_reg=l2_reg, strides=1)(B42)
        C44 = convolution(256, kernel_size=3, l2_reg=l2_reg, strides=1)(C43)
        C45 = convolution(256, kernel_size=3, l2_reg=l2_reg, strides=1)(C44)
        C45 = add([C41, C45])

        C51 = transposed_convolution(128, kernel_size=4, l2_reg=l2_reg, strides=2)(C45)
        C51 = add([C51, C34])
        C52 = Lambda(lambda x: K.concatenate([x,t2],axis=-1),output_shape=(nx/4,ny/4,256))(C51)
        B52 = Conv2D(128, (1, 1), padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), strides=1)(C52)
        C53 = convolution(128, kernel_size=3, l2_reg=l2_reg, strides=1)(B52)
        C54 = convolution(128, kernel_size=3, l2_reg=l2_reg, strides=1)(C53)
        C55 = convolution(128, kernel_size=3, l2_reg=l2_reg, strides=1)(C54)
        C55 = add([C51, C55])

        C61 = transposed_convolution(64, kernel_size=4, l2_reg=l2_reg, strides=2)(C55)
        C61 = add([C61, C24])
        C62 = Lambda(lambda x: K.concatenate([x,t3],axis=-1),output_shape=(nx/2,ny/2,128))(C61)
        B62 = Conv2D(128, (1, 1), padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), strides=1)(C62)
        C63 = convolution(64, kernel_size=3, l2_reg=l2_reg, strides=1)(B62)
        C64 = convolution(64, kernel_size=3, l2_reg=l2_reg, strides=1)(C63)
        C65 = convolution(64, kernel_size=3, l2_reg=l2_reg, strides=1)(C64)
        C65 = add([C61, C65])

        C71 = transposed_convolution(64, kernel_size=4, l2_reg=l2_reg, strides=2)(C65)
        C72 = convolution(64, kernel_size=4, l2_reg=l2_reg, strides=1)(C71)
        out = convolution(1, kernel_size=3, l2_reg=l2_reg, strides=1)(C72)

        return out, C43, C53, C63
    return fun

def mfbd(nx, ny, depth=5, l2_reg=0):
    """
    Deep residual network that keeps the size of the input throughout the whole network
    """

    inputs = [None] * depth
    estimation = [None] * depth
    for i in range(depth):
        inputs[i] = Input(shape=(nx,ny,1))

    t1 = K.zeros((10, int(nx/8), int(ny/8), 256))
    t2 = K.zeros((10, int(nx/4), int(ny/4), 128))
    t3 = K.zeros((10, int(nx/2), int(ny/2), 64))

    deconvolution = deconvolution_module(nx, ny, l2_reg)

    estimation[0] = inputs[0]

    for i in range(depth-1):
        estimation[i+1], t1, t2, t3 = deconvolution(estimation[i], inputs[i+1], t1, t2, t3)

    return Model(inputs=inputs, outputs=estimation[1:])