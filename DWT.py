import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer

'''
This file is taken directly from https://github.com/AureliiiieP/Keras-WaveletTransform/blob/master/models/DWT.py
'''

def normalize_data_format(value):
    if value is None:
        value = K.image_data_format()
    data_format = value.lower()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('The `data_format` argument must be one of '
                         '"channels_first", "channels_last". Received: ' +
                         str(value))
    return data_format

def dwt(x, data_format='channels_last'):

    """
    DWT (Discrete Wavelet Transform) function implementation according to
    "Multi-level Wavelet Convolutional Neural Networks"
    by Pengju Liu, Hongzhi Zhang, Wei Lian, Wangmeng Zuo
    https://arxiv.org/abs/1907.03128
    """

    if data_format == 'channels_last':
        x1 = x[:, 0::2, 0::2, :] #x(2i−1, 2j−1)
        x2 = x[:, 1::2, 0::2, :] #x(2i, 2j-1)
        x3 = x[:, 0::2, 1::2, :] #x(2i−1, 2j)
        x4 = x[:, 1::2, 1::2, :] #x(2i, 2j)

    elif data_format == 'channels_first':
        x1 = x[:, :, 0::2, 0::2] #x(2i−1, 2j−1)
        x2 = x[:, :, 1::2, 0::2] #x(2i, 2j-1)
        x3 = x[:, :, 0::2, 1::2] #x(2i−1, 2j)
        x4 = x[:, :, 1::2, 1::2] #x(2i, 2j)     

    x_LL = x1 + x2 + x3 + x4
    x_LH = -x1 - x3 + x2 + x4
    x_HL = -x1 + x3 - x2 + x4
    x_HH = x1 - x3 - x2 + x4

    if data_format == 'channels_last':
        return K.concatenate([x_LL,x_LH,x_HL,x_HH],axis=-1)
    elif data_format == 'channels_first':
        return K.concatenate([x_LL,x_LH,x_HL,x_HH],axis=1)


def iwt(x, data_format='channels_last'):
    """
    IWT (Inverse Wavelet Transfomr) function implementation according to
    "Multi-level Wavelet Convolutional Neural Networks"
    by Pengju Liu, Hongzhi Zhang, Wei Lian, Wangmeng Zuo
    https://arxiv.org/abs/1907.03128
    """
    if data_format == 'channels_last':

        x_LL = x[:, :, :, 0:x.shape[3]//4]
        x_LH = x[:, :, :, x.shape[3]//4:x.shape[3]//4*2]
        x_HL = x[:, :, :, x.shape[3]//4*2:x.shape[3]//4*3]
        x_HH = x[:, :, :, x.shape[3]//4*3:]

        x1 = (x_LL - x_LH - x_HL + x_HH)/4
        x2 = (x_LL - x_LH + x_HL - x_HH)/4
        x3 = (x_LL + x_LH - x_HL - x_HH)/4
        x4 = (x_LL + x_LH + x_HL + x_HH)/4 

        y1 = K.stack([x1,x3], axis=2)
        y2 = K.stack([x2,x4], axis=2)
        shape = K.shape(x)
        return K.reshape(K.concatenate([y1,y2], axis=-1), K.stack([shape[0], shape[1]*2, shape[2]*2, shape[3]//4]))

    elif data_format == 'channels_first':

        raise RuntimeError('WIP, please use "channels_last" instead.')

        x_LL = x[:, 0:x.shape[1]//4, :, :]
        x_LH = x[:, x.shape[1]//4:x.shape[1]//4*2, :, :]
        x_HL = x[:, x.shape[1]//4*2:x.shape[1]//4*3, :, :]
        x_HH = x[:, x.shape[1]//4*3:, :, :]

        x1 = (x_LL - x_LH - x_HL + x_HH)/4
        x2 = (x_LL - x_LH + x_HL - x_HH)/4
        x3 = (x_LL + x_LH - x_HL - x_HH)/4
        x4 = (x_LL + x_LH + x_HL + x_HH)/4 

        y1 = K.stack([x1,x3], axis=3)
        y2 = K.stack([x2,x4], axis=3)
        shape = K.shape(x)
        return K.reshape(K.concatenate([y1,y2], axis=1), K.stack([shape[0], shape[1]//4, shape[2]*2, shape[3]*2]))



class DWT_Pooling(Layer):
    """
    Custom Layer performing DWT pooling operation described in :

    "Multi-level Wavelet Convolutional Neural Networks"
    by Pengju Liu, Hongzhi Zhang, Wei Lian, Wangmeng Zuo
    https://arxiv.org/abs/1907.03128

    # Arguments :
        data_format (String): 'channels_first' or 'channels_last'

    # Input shape :
        If data_format='channels_last':
            4D tensor of shape: (batch_size, rows, cols, channels)
        If data_format='channels_first':
            4D tensor of shape: (batch_size, channels, rows, cols)

    # Output shape
        If data_format='channels_last':
            4D tensor of shape: (batch_size, rows/2, cols/2, channels*4)
        If data_format='channels_first':
            4D tensor of shape: (batch_size, channels*4, rows/2, cols/2)
    """

    def __init__(self, data_format=None,**kwargs):
        super(DWT_Pooling, self).__init__(**kwargs)
        self.data_format = normalize_data_format(data_format)

    def build(self, input_shape):
        super(DWT_Pooling, self).build(input_shape) 

    def call(self, x):
        return dwt(x, self.data_format)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            return (input_shape[0], input_shape[1]*4, input_shape[2]//2, input_shape[3]//2)
        elif self.data_format == 'channels_last':
            return (input_shape[0], input_shape[1]//2, input_shape[2]//2, input_shape[3]*4)


class IWT_UpSampling(Layer):
    """
    Custom Layer performing IWT upsampling operation described in :

    "Multi-level Wavelet Convolutional Neural Networks"
    by Pengju Liu, Hongzhi Zhang, Wei Lian, Wangmeng Zuo
    https://arxiv.org/abs/1907.03128

    # Arguments :
        data_format (String): 'channels_first' or 'channels_last'

    # Input shape :
        If data_format='channels_last':
            4D tensor of shape: (batch_size, rows, cols, channels)
        If data_format='channels_first':
            4D tensor of shape: (batch_size, channels, rows, cols)

    # Output shape
        If data_format='channels_last':
            4D tensor of shape: (batch_size, rows*2, cols*2, channels/4)
        If data_format='channels_first':
            4D tensor of shape: (batch_size, channels/4, rows*2, cols*2)
    """

    def __init__(self, data_format=None, **kwargs):
        super(IWT_UpSampling, self).__init__(**kwargs)
        self.data_format = normalize_data_format(data_format)

    def build(self, input_shape):
        super(IWT_UpSampling, self).build(input_shape) 

    def call(self, x):
        return iwt(x, self.data_format)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            return (input_shape[0], input_shape[1]//4, input_shape[2]*2, input_shape[3]*2)
        elif self.data_format == 'channels_last':
            return (input_shape[0], input_shape[1]*2, input_shape[2]*2, input_shape[3]//4)
