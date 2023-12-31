import argparse
from argparse import ArgumentParser
import glob
import cv2
import re
import os, glob, datetime
import numpy as np
import tensorflow as tf
import numpy as np
from scipy.io import loadmat
from skimage.transform import rescale
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

from tensorflow.keras.layers import  Input,Conv2D, BatchNormalization,Activation,Subtract, Reshape
from tensorflow.keras.models import Model, load_model
from tensorflow.python.keras.utils import conv_utils
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import Adam, SGD
#import data_generator as dg
import tensorflow.keras.backend as K
import skimage
from skimage.metrics import structural_similarity as ssim
from skimage.io import imread, imsave


##--------------------------------------------------------------------------------------------------------
class Hamiltonian_Conv2D(Conv2D):

    def __init__(self, filters, kernel_size, kernel_3=None, kernel_4=None, activation=None, **kwargs):

        self.rank = 2               # Dimension of the kernel
        self.num_filters = filters  # Number of filter in the convolution layer
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, self.rank, 'kernel_size')
        self.kernel_3 = kernel_3    # Weights from original potential
        self.kernel_4 = kernel_4    # Weights from interaction     

        super(Hamiltonian_Conv2D, self).__init__(self.num_filters, self.kernel_size, 
              activation=activation, use_bias=False, **kwargs)
        
    def build(self, input_shape):
        if K.image_data_format() == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                     'should be defined. Found `None`.')

        #don't use bias:
        self.bias = None

        #consider the layer built
        self.built = True


        # Define nabla operator
        weights_1 = tf.constant([[ 2.,-1., 0.],
                                 [-1., 4.,-1.],
                                 [ 0.,-1., 2.]])
        

        weights_1 = tf.reshape(weights_1 , [3,3, 1])
        weights_1 = tf.repeat(weights_1 , repeats=self.num_filters, axis=2)
        #print('kernel shape of weights_1:',weights_1.get_shape())

        # Define Weights for h^2/2m  (size should be same as the nabla operator)
        weights_2 = self.add_weight(shape=weights_1.get_shape(),
                                      initializer= 'Orthogonal',
                                      name='kernel_h^2/2m',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        #print('kernel shape of weights_2:',weights_2.get_shape())

        
        # Define the Hamiltonian kernel
        self.kernel = weights_1*weights_2 + self.kernel_3 + self.kernel_4
        #print('self.kernel',self.kernel.get_shape())

        self.built = True
        super(Hamiltonian_Conv2D, self).build(input_shape)

    # Do the 2D convolution using the Hamiltonian kernel
    def convolution_op(self, inputs, kernel):
        if self.padding == "causal":
            tf_padding = "VALID"  # Causal padding handled in `call`.
        elif isinstance(self.padding, str):
            tf_padding = self.padding.upper()
        else:
            tf_padding = self.padding


        return tf.nn.convolution(
            inputs,
            kernel,
            strides=list(self.strides),
            padding=tf_padding,
            dilations=list(self.dilation_rate),
            name=self.__class__.__name__,
        )

    def call(self, inputs):
        outputs = self.convolution_op(inputs, self.kernel)
        return outputs


      

# -------------------------------------------------------------------------------------------------------------------

def DIVA2D(depth,filters=64,image_channels=1, kernel_size=5, use_bnorm=True):
    layer_count = 0
    inpt = Input(shape=(None,None,image_channels),name = 'input'+str(layer_count))
    
    # Get the initial patches /initial_patches
    initial_patches = Conv2D(filters=filters, kernel_size=(kernel_size,kernel_size), strides=(1,1),kernel_initializer='Orthogonal', padding='same',name = 'initial_patches')(inpt)
    initial_patches = Activation('relu',name = 'initial_patch_acti')(initial_patches)
    #print(initial_patches.get_shape())

    # interaction layer
    inter = Conv2D(filters=filters, kernel_size=(kernel_size,kernel_size), strides=(1,1),kernel_initializer='Orthogonal', padding='same',name = 'interactions')(initial_patches)
    inter = Activation('relu',name = 'interaction_acti'+str(layer_count))(inter)
    #print(inter.get_shape())


    # Get contributions of the original potential in the Hamiltonian kernel
    ori_poten_kernel = tf.keras.layers.MaxPooling2D (pool_size=(21,21), strides=(15,15), padding='same', name = 'ori_poten_ker', data_format=None )(initial_patches)
    #print('ori_poten_kernel',ori_poten_kernel.get_shape())

    # Get contributions of the interactions in the Hamiltonian kernel
    inter_kernel = tf.keras.layers.MaxPooling2D (pool_size=(21,21), strides=(15,15), padding='same', name = 'inter_ker', data_format=None )(inter)
    #print('inter_kernel',inter_kernel.get_shape())


    # Get projection coefficients of the initial patches on the Hamiltonian kernel
    x = Hamiltonian_Conv2D(filters=filters, kernel_size=(kernel_size,kernel_size), kernel_3 = ori_poten_kernel, kernel_4 = inter_kernel, strides=(1,1), activation='relu',
                              kernel_initializer='Orthogonal', padding='same', name = 'proj_coef')(initial_patches)      
    
    #print('coef',x.get_shape())


    # Do Thresholding (depth depends on the noise intensity)
    for i in range(depth-2):
      layer_count += 1
      x = Conv2D(filters=filters, kernel_size=(kernel_size,kernel_size), strides=(1,1),kernel_initializer='Orthogonal', padding='same',use_bias = False,name = 'conv'+str(layer_count))(x)

      layer_count += 1
      x = BatchNormalization(axis=3, momentum=0.1,epsilon=0.0001, name = 'bn'+str(layer_count))(x) 
        #x = BatchNormalization(axis=3, momentum=0.0,epsilon=0.0001, name = 'bn'+str(layer_count))(x)
      
      # Thresholding
      x = Activation('relu',name = 'Thresholding'+str(layer_count))(x)  

    # Inverse projection
    x = Conv2D(filters=image_channels, kernel_size=(kernel_size,kernel_size), strides=(1,1), kernel_initializer='Orthogonal',padding='same',use_bias = False,name = 'inv_trans')(x)

    x = Subtract(name = 'subtract')([inpt, x])

    model = Model(inputs=inpt, outputs=x)
    
    return model

##----------------------------------------------------------------------------------------------------------------------


def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir,'model_*.hdf5'))  # get name list of all .hdf5 files
    #file_list = os.listdir(save_dir)
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).hdf5.*",file_)
            #print(result[0])
            epochs_exist.append(int(result[0]))
        initial_epoch=max(epochs_exist)   
    else:
        initial_epoch = 0
    return initial_epoch


def show(x,title=None,cbar=False,figsize=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    plt.imshow(x,interpolation='nearest',cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()


def log(*args,**kwargs):
     print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"),*args,**kwargs)

def lr_schedule(epoch):
    initial_lr = args.lr
    if epoch<=20:
        lr = initial_lr
    elif epoch<=30:
        lr = initial_lr/10
    elif epoch<=40:
        lr = initial_lr/20
    else:
        lr = initial_lr/20
    log('current learning rate is %2.8f' %lr)
    return lr

def train_datagen(epoch_iter=2000,epoch_num=5,batch_size=128,data_dir='./data/training_set'):
  import random
  import math

  while(True):
      n_count = 0
      if n_count == 0:
          #print(n_count)
          xs, ys = datagenerator(data_dir)
          assert len(xs)%batch_size ==0, \
          log('make sure the last iteration has a full batchsize, this is important if you use batch normalization!')

	  # Normalized the data sets
          xs = xs.astype('float32')/255.0   # if the images are not normalized then divide by 255
          ys = ys.astype('float32')/255.0   # if the images are not normalized then divide by 255

          indices = list(range(xs.shape[0]))
          n_count = 1
      for _ in range(epoch_num):
          np.random.shuffle(indices)    # shuffle
          for i in range(0, len(indices), batch_size):
              batch_x = xs[indices[i:i+batch_size]]
              batch_y = ys[indices[i:i+batch_size]]
              
            #  random.seed(1)
              #noise =  np.random.normal(0, args.sigma/255.0, batch_x.shape)
              #batch_y = batch_x + noise 
              yield batch_y, batch_x



patch_size, stride = 40, 10
aug_times = 1
num_noise_realiza = 2
scales = [1] #, 0.9, 0.8, 0.7
batch_size = 8


def data_aug(img, mode=0):

    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img)
    elif mode == 3:
        return np.flipud(np.rot90(img))
    elif mode == 4:
        return np.rot90(img, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))



def noise_aug(mode=0):
    # noise level augmentation
    if mode == 0:
         sigma = 2
    elif mode == 1:
         sigma = 4
    elif mode == 2:
         sigma = 6
    elif mode == 3:
         sigma = 8
    elif mode == 4:
         sigma = 10
    elif mode == 5:
         sigma = 12
    elif mode == 6:
         sigma = 14
    elif mode == 7:
         sigma = 16
    elif mode == 8:
         sigma = 18
    elif mode == 9:
         sigma = 20

    return sigma


def gen_patches(file_name, sigma):

    # read image
    clean_img = cv2.imread(file_name, 0)  # load from .png file
    # I = loadmat(file_name)         	  # load from .mat file
    # clean_img = I['img']            	  # load from .mat file
    h, w = clean_img.shape

    noise_sigma = sigma
    noise = np.random.normal(0, noise_sigma, clean_img.shape)
    img = clean_img + noise
    
    # showing image  
    #fig = plt.figure(figsize=(12, 4))
    #plt.subplot(1, 2, 1)
    #plt.imshow(clean_img, cmap='gray')
    #plt.colorbar()
    #plt.title("Ground Truth")
        
    #plt.subplot(1, 2, 2)
    #plt.imshow(img, cmap='gray')
    #plt.colorbar()
    #plt.title("Noisy image")


    patches = []
    clean_patches = []

    for s in scales:
        h_scaled, w_scaled = int(h*s), int(w*s)
        img_scaled = img 
        clean_img_scaled = clean_img
        #img_scaled = cv2.resize(img, (h_scaled,w_scaled), interpolation=cv2.INTER_CUBIC)

        # extract patches
        for i in range(0, h_scaled-patch_size+1, stride):
            for j in range(0, w_scaled-patch_size+1, stride):
                    x = img_scaled[i:i+patch_size, j:j+patch_size]
                    clean_x = clean_img_scaled[i:i+patch_size, j:j+patch_size]

                    #patches.append(x)        
                    # data augmentation with different rotation
                    for k in range(0, aug_times):
                        mode_k=np.random.randint(0,8)
                        x_aug = data_aug(x, mode=mode_k)                        
                        clean_x_aug = data_aug(clean_x, mode=mode_k)
                        
                        patches.append(x_aug)
                        clean_patches.append(clean_x_aug)
                
    return clean_patches, patches


def datagenerator(data_dir='data/train5',verbose=False):
    
    file_list = glob.glob(data_dir+'/*.png')     # get name list of all .png files
    # file_list = glob.glob(data_dir+'/*.mat')   # get name list of all .mat files

    # initrialize
    data = []
    data_clean = []

    # generate patches
    for i in range(len(file_list)):

        # Get a noise level
        sigma = 15			  # For a fixed noise level
        sigma = noise_aug(mode=np.random.randint(0, 6))   # For a range of different noise level

        #print("Noise level:",sigma)

        # data augmentation with different noise realization
        for repeat_aug in range(0, num_noise_realiza):
           
            clean_patch, patch = gen_patches(file_list[i], sigma)

            data.append(patch)
            data_clean.append(clean_patch)
        
        #if verbose:
        #   print(str(i+1)+'/'+ str(len(file_list)) + ' is done ^_^')


    # do for noisy data
    data = np.array(data, dtype='float32')
    data = data.reshape((data.shape[0]*data.shape[1],data.shape[2],data.shape[3],1))
    discard_n = len(data)-len(data)//batch_size*batch_size;
    data = np.delete(data,range(discard_n),axis = 0)

    # do for clean data
    data_clean = np.array(data_clean, dtype='float32')
    data_clean = data_clean.reshape((data_clean.shape[0]*data_clean.shape[1],data_clean.shape[2],data_clean.shape[3],1))
    discard_n = len(data_clean)-len(data_clean)//batch_size*batch_size;
    data_clean = np.delete(data_clean,range(discard_n),axis = 0)
    print('^_^-training data finished-^_^')
    print(data.shape)
    #print(data_clean.shape)
    
    return data_clean, data


##---------------------------------------------------------------------------------------------------------


# define loss
def sum_squared_error(y_true, y_pred):
    #return K.mean(K.square(y_pred - y_true), axis=-1)
    #return K.sum(K.square(y_pred - y_true), axis=-1)/2
    return K.sum(K.square(y_pred - y_true))/2
