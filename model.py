import tensorflow as tf
from tensorflow import keras
import numpy as np
from DWT import DWT_Pooling, IWT_UpSampling

def conv_block(inputs: tf.Tensor,
               num_filters: int,
               kernel_size: int,
               num_convs: int = 2,
               ) -> tf.Tensor:
    
    outs = inputs
    for i in range(num_convs):
        outs = keras.layers.Conv2D(num_filters, kernel_size, padding = 'same')(outs)
        outs = keras.layers.LeakyReLU(0.01)(outs)
    
    shortcut = keras.layers.Conv2D(num_filters, (1,1), padding = 'same')(inputs)
    shortcut = keras.layers.LeakyReLU(0.01)(shortcut)
    outs = keras.layers.Add()([shortcut, outs])
    
    return outs

def upconv_block(inputs: tf.Tensor,
               num_filters: int,
               kernel_size: int,
               skip_connection: tf.Tensor,
               num_convs: int = 2,
               use_DWT: bool = False
               ) -> tf.Tensor:

    if use_DWT:
        outs = IWT_UpSampling()(inputs)
    else:
        outs = keras.layers.Conv2DTranspose(num_filters, kernel_size, strides=2, padding = 'same')(inputs)

    if skip_connection is not None:
        outs = keras.layers.LeakyReLU(0.01)(outs)
        outs = keras.layers.concatenate([skip_connection, outs], axis = 3)

    shortcut = shortcut = keras.layers.Conv2D(num_filters, (1,1), padding = 'same')(outs)
    shortcut = keras.layers.LeakyReLU(0.01)(shortcut)
    for i in range(num_convs):
        outs = keras.layers.Conv2D(num_filters, 3, padding = 'same')(outs)
        outs = keras.layers.LeakyReLU(0.01)(outs)
    
    outs = keras.layers.Add()([shortcut, outs])

    return outs

def get_model(num_blocks: int = 4,
              num_convs: int = 2,
              input_shape: tuple = (128, 128, 3),
              use_DWT: bool = False
              ) -> keras.Model:
    
    inputs = keras.Input(shape=input_shape)
    outs = inputs
    conv_block_outs = []
    
    for i in range(num_blocks-1):
        outs = conv_block(outs, 16*(2**i), 3, num_convs)
        conv_block_outs.append(outs)
        if use_DWT:
            outs = DWT_Pooling()(outs)
        else:
            outs = keras.layers.MaxPooling2D(pool_size=(2,2))(outs)

    outs = conv_block(outs, 16*2**(num_blocks-1), 3)

    for i in range(num_blocks-1):
        outs = upconv_block(outs, 16*2**(num_blocks-i-1), 3, conv_block_outs.pop(), num_convs)

    #outs = upconv_block(outs, 16, 3, None, num_convs)
    outs = keras.layers.Conv2D(3, 1, activation='sigmoid')(outs)

    model = tf.keras.Model(inputs = inputs, outputs = outs)
    return model
    










\

