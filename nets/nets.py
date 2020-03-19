# -*- coding: utf-8 -*-

import numpy as np
import h5py
import os.path
from sklearn.metrics import roc_auc_score
 

from keras.models import Sequential, Model
from keras.layers import *
from keras.layers import Dense
from keras import backend as K
from keras.utils.np_utils import to_categorical
from keras.layers.advanced_activations import PReLU,ELU,LeakyReLU

from keras.models import Model
from keras.layers import Dense, Activation, Input, Reshape
from keras.layers import Conv1D, Flatten, Dropout
from keras.optimizers import SGD, Adam
from keras.layers.normalization import BatchNormalization
from scipy.io import loadmat
 

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from scipy.optimize import minimize_scalar

import pandas as pd
import resnet2

from keras import regularizers


from keras.layers.core import Layer
from keras.engine import InputSpec
from keras import backend as K
from keras import initializers
import tensorflow as tf
from keras.engine import *

try:
    from keras import initializations
except ImportError:
    from keras import initializers as initializations 
    
    
class DenseSN(Dense):
    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.u = self.add_weight(shape=tuple([1, self.kernel.shape.as_list()[-1]]),
                                 initializer=initializers.RandomNormal(0, 1),
                                 name='sn',
                                 trainable=False)
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True
        
    def call(self, inputs, training=None):
        def _l2normalize(v, eps=1e-12):
            return v / (K.sum(v ** 2) ** 0.5 + eps)
        def power_iteration(W, u):
            _u = u
            _v = _l2normalize(K.dot(_u, K.transpose(W)))
            _u = _l2normalize(K.dot(_v, W))
            return _u, _v
        W_shape = self.kernel.shape.as_list()
        #Flatten the Tensor
        W_reshaped = K.reshape(self.kernel, [-1, W_shape[-1]])
        _u, _v = power_iteration(W_reshaped, self.u)
        #Calculate Sigma
        sigma=K.dot(_v, W_reshaped)
        sigma=K.dot(sigma, K.transpose(_u))
        #normalize it
        W_bar = W_reshaped / sigma
        #reshape weight tensor
        if training in {0, False}:
            W_bar = K.reshape(W_bar, W_shape)
        else:
            with tf.control_dependencies([self.u.assign(_u)]):
                 W_bar = K.reshape(W_bar, W_shape)  
        output = K.dot(inputs, W_bar)
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output 
    
    
class ZscoreLayer(Layer):

    def __init__(self, weights=None, **kwargs):
        self.mean_init = initializers.Zeros()
        self.std_init = initializers.Ones()
        self.initial_weights = weights
        super(ZscoreLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        # 1：InputSpec(dtype=None, shape=None, ndim=None, max_ndim=None, min_ndim=None, axes=None)
        #Docstring:     
        #Specifies the ndim, dtype and shape of every input to a layer.
        #Every layer should expose (if appropriate) an `input_spec` attribute:a list of instances of InputSpec (one per input tensor).
        #A None entry in a shape is compatible with any dimension
        #A None shape is compatible with any shape.

        # 2:self.input_spec: List of InputSpec class instances
        # each entry describes one required input:
        #     - ndim
        #     - dtype
        # A layer with `n` input tensors must have
        # an `input_spec` of length `n`.

        shape = (int(input_shape[1]),)

        # Compatibility with TensorFlow >= 1.0.0
        self.mean = K.variable(self.mean_init(shape), name='{}_mean'.format(self.name))
        self.std = K.variable(self.std_init(shape), name='{}_std'.format(self.name))

        self.non_trainable_weights = [self.mean, self.std]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
#             del self.initial_weights
            
        super(ZscoreLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, mask=None):
        out = (x - K.expand_dims(self.mean,axis=0))/K.expand_dims(self.std,axis=0)
        return out
    
    def get_config(self):
#         config = {"weights": self.initial_weights}
        config = {}
        base_config = super(ZscoreLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
  
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    

class ZscoreLayerRNN(Layer):
 
    def __init__(self, weights=None, **kwargs):
        self.mean_init = initializers.Zeros()
        self.std_init = initializers.Ones()
        self.initial_weights = weights
        super(ZscoreLayerRNN, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        # 1：InputSpec(dtype=None, shape=None, ndim=None, max_ndim=None, min_ndim=None, axes=None)
        #Docstring:     
        #Specifies the ndim, dtype and shape of every input to a layer.
        #Every layer should expose (if appropriate) an `input_spec` attribute:a list of instances of InputSpec (one per input tensor).
        #A None entry in a shape is compatible with any dimension
        #A None shape is compatible with any shape.

        # 2:self.input_spec: List of InputSpec class instances
        # each entry describes one required input:
        #     - ndim
        #     - dtype
        # A layer with `n` input tensors must have
        # an `input_spec` of length `n`.

        shape = (int(input_shape[2]),)

        # Compatibility with TensorFlow >= 1.0.0
        self.mean = K.variable(self.mean_init(shape), name='{}_mean'.format(self.name))
        self.std = K.variable(self.std_init(shape), name='{}_std'.format(self.name))

        self.non_trainable_weights = [self.mean, self.std]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
#             del self.initial_weights
            
        super(ZscoreLayerRNN, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, mask=None):
        out = (x - K.expand_dims(K.expand_dims(self.mean,axis=0),axis=1))/K.expand_dims(K.expand_dims(self.std,axis=0),axis=1)
        return out
    
    def get_config(self):
#         config = {"weights": self.initial_weights}
        config = {}
        base_config = super(ZscoreLayerRNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
  
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    
 
def swish(x):
    y = Activation('sigmoid')(x)
    return multiply([x,y])


def outer_swish(x):
    
    y=Lambda(outer_swish_sub)(x)
    
    return y

def outer_swish_sub(x):
    y = K.sigmoid(x)
    outerProduct = K.expand_dims(x,axis=2) * K.expand_dims(y,axis=1)
    return K.logsumexp(outerProduct,axis=1)

 
def input_model_fc(hidden_dim,input_shape):
    inputs = Input(input_shape, name="input")  
    x = Dense(hidden_dim, activation='linear')(inputs)  
    x = BatchNormalization()(x)
    x0 = swish(x)
#     flat = x0
    flat = Dropout(0.5)(x0)
    model_input = Model(inputs, flat)
     
    shape_before_flatten = x0.shape.as_list()[1:] # [1:] to skip None
    shape_flatten = np.prod(shape_before_flatten) # value of shape in the non-batch dimension
    
#     print shape_flatten
    
    return model_input,inputs, flat,shape_flatten

def input_model_fc_2layers(hidden_dim,input_shape):
    inputs = Input(input_shape, name="input")  
    x = Dense(hidden_dim, activation='linear')(inputs) 
    x = BatchNormalization()(x)
    x0 = swish(x)
#     flat = x0
    x = Dropout(0.25)(x0)
    x = Dense(hidden_dim, activation='linear')(x)  
    x = BatchNormalization()(x)
    x0 = swish(x)
#     flat = x0
    flat = Dropout(0.25)(x0)
    model_input = Model(inputs, flat)
     
    shape_before_flatten = x0.shape.as_list()[1:] # [1:] to skip None
    shape_flatten = np.prod(shape_before_flatten) # value of shape in the non-batch dimension
    
#     print shape_flatten
    
    return model_input,inputs, flat,shape_flatten
  
def input_model_simpleCNN(hidden_dim,input_shape): 
    inputs = Input(input_shape, name="input")
    conv1 = Convolution2D(32, 3, 3, activation='relu', name="conv1")(inputs)
    conv2 = Convolution2D(32, 3, 3, activation='relu', name="conv2")(conv1)
    pool = MaxPooling2D((2, 2), name="pool1")(conv2)
    drop = Dropout(0.25, name="drop")(pool)
    flat = Flatten(name="flat")(drop)
    model_input = Model(inputs, flat)
    
     
    shape_before_flatten = drop.shape.as_list()[1:] # [1:] to skip None
    shape_flatten = np.prod(shape_before_flatten) # value of shape in the non-batch dimension
    
#     print shape_flatten

    return model_input,inputs, flat,shape_flatten
 
def input_model_resnet(hidden_dim,input_shape): 
    model_input,inputs, flat,shape_flatten = resnet2.ResnetBuilder.build_resnet_18((input_shape[2],input_shape[0],input_shape[1]), 10)
    print shape_flatten

    return model_input,inputs, flat,shape_flatten
 
def input_model_simpleCNN_convout(hidden_dim,input_shape): 
    inputs = Input(input_shape, name="input")
    conv1 = Convolution2D(hidden_dim, 3, 3, activation='relu', name="conv1")(inputs)
    conv2 = Convolution2D(hidden_dim, 3, 3, activation='relu', name="conv2")(conv1)
    pool = MaxPooling2D((2, 2), name="pool1")(conv2)
    drop = Dropout(0.25, name="drop")(pool)
#     flat = Flatten(name="flat")(drop)
    model_input = Model(inputs, drop)
     
    shape_before_flatten = drop.shape.as_list()[1:] # [1:] to skip None
#     shape_flatten = np.prod(shape_before_flatten) # value of shape in the non-batch dimension
    
#     print shape_flatten

    return model_input,inputs, drop,shape_before_flatten




def input_model_simpleCNNv2(hidden_dim,input_shape):
    inputs = Input(input_shape, name="input")
    conv1 = Convolution2D(hidden_dim, (3, 3), activation='relu', name='conv1', padding='same')(inputs)
    conv2 = Convolution2D(hidden_dim, (3, 3), activation='relu', name='conv2', padding='same')(conv1)
    pool = MaxPooling2D((2, 2), name="pool1")(conv2)
    drop = Dropout(0.25, name="drop")(pool)
    flat = Flatten(name="flat")(drop)
    model_input = Model(inputs, flat)
    
    
    shape_before_flatten = drop.shape.as_list()[1:] # [1:] to skip None
    shape_flatten = np.prod(shape_before_flatten) # value of shape in the non-batch dimension
    
#     print shape_flatten

    return model_input,inputs, flat,shape_flatten


def input_model_simpleCNNv3(hidden_dim,input_shape):
    inputs = Input(input_shape, name="input")
    conv1 = Convolution2D(hidden_dim, (3, 3), activation='relu', padding='same')(inputs)
    conv2 = Convolution2D(hidden_dim, (3, 3), activation='relu', padding='same')(conv1)
    pool = MaxPooling2D((2, 2))(conv2)
    drop = Dropout(0.25)(pool)
    conv1 = Convolution2D(hidden_dim*2, (3, 3), activation='relu', padding='same')(drop)
    conv2 = Convolution2D(hidden_dim*2, (3, 3), activation='relu', padding='same')(conv1)
    pool = MaxPooling2D((2, 2))(conv2)
    drop = Dropout(0.25)(pool)
    flat = Flatten(name="flat")(drop)
    model_input = Model(inputs, flat)
    
    
    shape_before_flatten = drop.shape.as_list()[1:] # [1:] to skip None
    shape_flatten = np.prod(shape_before_flatten) # value of shape in the non-batch dimension
    
#     print shape_flatten

    return model_input,inputs, flat,shape_flatten


def input_model_simpleCNNv4(hidden_dim,input_shape):
    inputs = Input(input_shape, name="input")
    conv1 = Convolution2D(32, 3, 3, activation='relu', name="conv1")(inputs)
    conv2 = Convolution2D(32, 3, 3, activation='relu', name="conv2")(conv1)
    pool = MaxPooling2D((2, 2), name="pool1")(conv2)
    drop = Dropout(0.25, name="drop")(pool)
    x = Flatten(name="flat")(drop)
    x = BatchNormalization()(x)
    x = Dense(hidden_dim, activation='linear')(x) 
    x = BatchNormalization()(x)
    x = swish(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(hidden_dim, activation='linear')(x) 
    x = BatchNormalization()(x)
    x0 = swish(x)
    flat = Dropout(0.5)(x0)
    model_input = Model(inputs, flat)
    
    
    shape_before_flatten = x0.shape.as_list()[1:] # [1:] to skip None
    shape_flatten = np.prod(shape_before_flatten) # value of shape in the non-batch dimension
    
#     print shape_flatten

    return model_input,inputs, flat,shape_flatten


def input_model_simpleCNNv5(hidden_dim,input_shape):
    inputs = Input(input_shape, name="input")
    x = Convolution2D(32, 3, 3, activation='linear')(inputs)
    x = BatchNormalization()(x)
    x = swish(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    for i in range(2):
        x = Convolution2D(32*2**i, 3, 3, activation='linear')(x)
        x = BatchNormalization()(x)
        x = swish(x) 
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)
    flat = Flatten()(x)
    model_input = Model(inputs, flat)
    
    
    shape_before_flatten = x.shape.as_list()[1:] # [1:] to skip None
    shape_flatten = np.prod(shape_before_flatten) # value of shape in the non-batch dimension
    
#     print shape_flatten

    return model_input,inputs, flat,shape_flatten


def input_model_simpleCNNv6(hidden_dim,input_shape):
    inputs = Input(input_shape, name="input")
    x = Convolution2D(32, (3, 3), activation='linear')(inputs)
    x = BatchNormalization()(x)
    x = swish(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    for i in range(2):
        for _ in range(2):
            x = Convolution2D(32*2**i, (3, 3), activation='linear', padding='same')(x)
            x = BatchNormalization()(x)
            x = swish(x) 
            x = Dropout(0.25)(x)
        x = MaxPooling2D((2, 2))(x)
        
    flat = Flatten()(x)
    model_input = Model(inputs, flat)
    
    
    shape_before_flatten = x.shape.as_list()[1:] # [1:] to skip None
    shape_flatten = np.prod(shape_before_flatten) # value of shape in the non-batch dimension
    
#     print shape_flatten

    return model_input,inputs, flat,shape_flatten



def input_model_simpleCNNv7_convout(hidden_dim,input_shape,depth=3):
    inputs = Input(input_shape, name="input")
    x = Conv2D(hidden_dim, kernel_size=3, strides=(1, 1), padding='same', activation='linear', kernel_initializer="he_normal")(inputs)
    x = BatchNormalization()(x)
    x0 = swish(x)
    flat = Dropout(1./(depth+1))(x0)
    for i in range(depth):
        x = Convolution2D(hidden_dim, kernel_size=3, strides=(1, 1), padding='same', activation='linear', kernel_initializer="he_normal")(x)
        x = BatchNormalization()(x)
        x0 = swish(x)
        flat0 = Dropout(1./(depth+1))(x0)
        flat = add([flat,flat0])
 
    model_input = Model(inputs, flat)
    
    
    shape_before_flatten = x.shape.as_list()[1:] # [1:] to skip None
#     shape_flatten = np.prod(shape_before_flatten) # value of shape in the non-batch dimension
    
#     print shape_flatten

    return model_input,inputs, flat,shape_before_flatten


def input_model_simpleCNNv8_convout(hidden_dim,input_shape,depth=3):
    inputs = Input(input_shape, name="input")
    x = Conv2D(hidden_dim, kernel_size=3, strides=(1, 1), padding='same', activation='relu')(inputs)
    x = Conv2D(hidden_dim, kernel_size=3, strides=(1, 1), padding='same', activation='relu')(x)
 
    flat = Dropout(0.25)(x)
     
    model_input = Model(inputs, flat)
    
    
    shape_before_flatten = x.shape.as_list()[1:] # [1:] to skip None
#     shape_flatten = np.prod(shape_before_flatten) # value of shape in the non-batch dimension
    
#     print shape_flatten

    return model_input,inputs, flat,shape_before_flatten
 
def fbnq(n):
    return ((1.+np.sqrt(5.))**n-(1.-np.sqrt(5))**n)/(2.**n*np.sqrt(5.))

# 
def input_model_fc_deep_fbnq(hidden_dim,input_shape,depth):
    inputs = Input(input_shape, name="input") 
    x = Dense(hidden_dim, activation='linear')(inputs) 
    x = BatchNormalization()(x)
    x0 = swish(x)
    flat = Dropout(1./(depth+1))(x0)
    flat = Lambda(lambda x: x /fbnq(depth+1))(flat)
    for i in range(depth):
        x = Dense(hidden_dim, activation='linear')(flat) 
#         x = BatchNormalization()(x)
        x0 = swish(x)
        flat0 = Dropout(1./(depth+1))(x0)
        flat0 = Lambda(lambda x: x /fbnq(depth-i))(flat0)
        flat = add([flat,flat0])
    
    
    model_input = Model(inputs, flat)
    
    
    shape_before_flatten = x0.shape.as_list()[1:] # [1:] to skip None
    shape_flatten = np.prod(shape_before_flatten) # value of shape in the non-batch dimension
    
#     print shape_flatten
    
    return model_input,inputs, flat,shape_flatten

# 
def input_model_fc_deep_exp(hidden_dim,input_shape,depth):
    inputs = Input(input_shape, name="input") 
    x = Dense(hidden_dim, activation='linear')(inputs) 
    x = BatchNormalization()(x)
    x0 = swish(x)
    flat = Dropout(1./(depth+1))(x0)
    flat0 = Lambda(lambda x: x /float(np.math.factorial(i+1)) )(flat0)
    for i in range(depth):
        x = Dense(hidden_dim, activation='linear')(flat) 
#         x = BatchNormalization()(x)
        x0 = swish(x)
        flat0 = Dropout(1./(depth+1))(x0)
        flat = add([flat,flat0])
    
    
    model_input = Model(inputs, flat)
    
    
    shape_before_flatten = x0.shape.as_list()[1:] # [1:] to skip None
    shape_flatten = np.prod(shape_before_flatten) # value of shape in the non-batch dimension
    
#     print shape_flatten
    
    return model_input,inputs, flat,shape_flatten
 

def ada_eye(mat):
    col = K.tf.reduce_sum(mat, 1)
    col = K.tf.ones_like(col)
    return K.tf.diag(col)

def orth_reg(W):
    WT = K.transpose(W)
    X = K.dot(WT,W)
    X = X *(1.- ada_eye(X))
    return 1e-3 * K.sum(X*X)
 


# 
def input_model_fc_deep(hidden_dim,input_shape,depth,inputs=None,input_layer=None,flag_addBN = False,flag_include_zscore=False,zscore_weights=None,flag_outer_swish=False,flag_nodropout=False,flag_orth_reg=False,kernel_l2=0.0,flag_orth_init=False,flag_skip_input=False,flag_SN=False,flag_LMH=False,activity_l1 = 0.0,flag_BN=True):
   
    if flag_orth_reg:
        regu = orth_reg
    else:
        regu = regularizers.l2(kernel_l2)
    activity_regu = regularizers.l1(activity_l1)   
        
    if flag_orth_init:
        init = initializers.Orthogonal()
    else:
        init = 'glorot_uniform'
        
    if flag_SN:
        Dense_final = DenseSN
    else:
        Dense_final = Dense
    
    
    if inputs == None:
        inputs = Input(input_shape, name="input") 
        if flag_include_zscore:
            x = ZscoreLayer(weights=zscore_weights)(inputs)
            if flag_skip_input:
                skip_input = x
            x = Dense_final(hidden_dim, activation='linear',kernel_regularizer=regu,kernel_initializer=init,activity_regularizer=activity_regu)(x)
        else:
            x = Dense_final(hidden_dim, activation='linear',kernel_regularizer=regu,kernel_initializer=init,activity_regularizer=activity_regu)(inputs) 
            if flag_skip_input:
                skip_input = inputs
    else:
        if flag_include_zscore:
            x = ZscoreLayer(weights=zscore_weights)(input_layer)
            if flag_skip_input:
                skip_input = x
            x = Dense_final(hidden_dim, activation='linear',kernel_regularizer=regu,kernel_initializer=init,activity_regularizer=activity_regu)(x)
        else:
            x = Dense_final(hidden_dim, activation='linear',kernel_regularizer=regu,kernel_initializer=init,activity_regularizer=activity_regu)(input_layer) 
            if flag_skip_input:
                skip_input = input_layer
    if flag_BN:
        x = BatchNormalization()(x)
    if flag_outer_swish:
        x0 = outer_swish(x)
    else:
        x0 = swish(x)
    if flag_nodropout:
        flat = x0
    else:
        flat = Dropout(1./(depth+1))(x0)
        
    LMH_list = [flat]
    for i in range(depth):
        if flag_skip_input:
            concat_x = concatenate([flat,skip_input])
            x = Dense_final(hidden_dim, activation='linear',kernel_regularizer=regu,kernel_initializer=init,activity_regularizer=activity_regu)(concat_x) 
        else:
            x = Dense_final(hidden_dim, activation='linear',kernel_regularizer=regu,kernel_initializer=init,activity_regularizer=activity_regu)(flat) 
        if flag_BN:
            x = BatchNormalization()(x)
        if flag_outer_swish:
            x0 = outer_swish(x)
        else:
            x0 = swish(x)
        if flag_nodropout:
            flat0 = x0
        else:
            flat0 = Dropout(1./(depth+1))(x0)
        flat = add([flat,flat0])
        
        if i in [0,depth/2,depth-1]:
            LMH_list.append(flat)
        
        
    if flag_addBN:
        flat = BatchNormalization()(flat)
        
    if flag_LMH:
        flat = concatenate(LMH_list)
#     if flag_skip_input:
#         flat = concatenate([flat,skip_input])
    
    model = Model(inputs, flat)
    opt = Adam(lr=1e-3,beta_1=0.5,beta_2=0.999) #SGD(lr=lr)   
    model.compile(loss='mse', optimizer=opt)  
    
    
    shape_before_flatten = flat.shape.as_list()[1:] # [1:] to skip None
    shape_flatten = np.prod(shape_before_flatten) # value of shape in the non-batch dimension
    
#     print shape_flatten
    
    return model,inputs, flat,shape_flatten

def input_model_lstm_deep(hidden_dim,input_shape,depth,inputs=None,input_layer=None,flag_addBN = False,flag_include_zscore=False,zscore_weights=None,flag_outer_swish=False,flag_nodropout=False):
    if inputs == None:
        inputs = Input(input_shape, name="input") 
        if flag_include_zscore:
            x = ZscoreLayerRNN(weights=zscore_weights)(inputs)
            x = LSTM(hidden_dim)(x)
            x = Dense(hidden_dim, activation='linear')(x)
        else:
            x = LSTM(hidden_dim)(inputs)
            x = Dense(hidden_dim, activation='linear')(x) 
    else:
        if flag_include_zscore:
            x = ZscoreLayerRNN(weights=zscore_weights)(input_layer)
            x = LSTM(hidden_dim)(x)
            x = Dense(hidden_dim, activation='linear')(x)
        else:
            x = LSTM(hidden_dim)(input_layer)
            x = Dense(hidden_dim, activation='linear')(x) 
    x = BatchNormalization()(x)
    if flag_outer_swish:
        x0 = outer_swish(x)
    else:
        x0 = swish(x)
    if flag_nodropout:
        flat = x0
    else:
        flat = Dropout(1./(depth+1))(x0)
    for i in range(depth):
        x = Dense(hidden_dim, activation='linear')(flat) 
        x = BatchNormalization()(x)
        if flag_outer_swish:
            x0 = outer_swish(x)
        else:
            x0 = swish(x)
        if flag_nodropout:
            flat0 = x0
        else:
            flat0 = Dropout(1./(depth+1))(x0)
        flat = add([flat,flat0])
    if flag_addBN:
        flat = BatchNormalization()(flat)
    
    model = Model(inputs, flat)
    opt = Adam(lr=1e-3,beta_1=0.5,beta_2=0.999) #SGD(lr=lr)   
    model.compile(loss='mse', optimizer=opt)  
    
    
    shape_before_flatten = x0.shape.as_list()[1:] # [1:] to skip None
    shape_flatten = np.prod(shape_before_flatten) # value of shape in the non-batch dimension
    
#     print shape_flatten
    
    return model,inputs, flat,shape_flatten
 

def input_model_fc_deep_nobn(hidden_dim,input_shape,depth,inputs=None,input_layer=None,flag_addBN = False):
    if inputs == None:
        inputs = Input(input_shape, name="input") 
        x = Dense(hidden_dim, activation='linear')(inputs) 
    else:
        x = Dense(hidden_dim, activation='linear')(input_layer) 
#     x = BatchNormalization()(x)
    x0 = swish(x)
    flat = Dropout(1./(depth+1))(x0)
    for i in range(depth):
        x = Dense(hidden_dim, activation='linear')(flat) 
#         x = BatchNormalization()(x)
        x0 = swish(x)
        flat0 = Dropout(1./(depth+1))(x0)
        flat = add([flat,flat0])
#     if flag_addBN:
#         flat = BatchNormalization()(flat)
    
    model_input = Model(inputs, flat)
    
    
    shape_before_flatten = x0.shape.as_list()[1:] # [1:] to skip None
    shape_flatten = np.prod(shape_before_flatten) # value of shape in the non-batch dimension
    
#     print shape_flatten
    
    return model_input,inputs, flat,shape_flatten

 
def decoder_model_fc_deep(input_dim,hidden_dim,depth):
    inputs = Input((input_dim,)) 
    x = Dense(hidden_dim, activation='linear')(inputs) 
    x = BatchNormalization()(x)
    x0 = swish(x)
    flat = Dropout(1./(depth+1))(x0)
    for i in range(depth):
        x = Dense(hidden_dim, activation='linear')(flat) 
        x = BatchNormalization()(x)
        x0 = swish(x)
        flat0 = Dropout(1./(depth+1))(x0)
        flat = add([flat,flat0])
    
    
    model_input = Model(inputs, flat)
    
    
    shape_before_flatten = x0.shape.as_list()[1:] # [1:] to skip None
    shape_flatten = np.prod(shape_before_flatten) # value of shape in the non-batch dimension
    
#     print shape_flatten
    
    return model_input,inputs, flat,shape_flatten



# 
def input_model_fc_deep_nodrop(hidden_dim,input_shape,depth):
    inputs = Input(input_shape, name="input") 
    x = Dense(hidden_dim, activation='linear')(inputs) 
    x = BatchNormalization()(x)
    x0 = swish(x)
    flat = Dropout(1./(depth+1))(x0)
    for i in range(depth):
        x = Dense(hidden_dim, activation='linear')(flat) 
        x = BatchNormalization()(x)
        flat0 = swish(x)
#         flat0 = Dropout(1./(depth+1))(x0)
        flat = add([flat,flat0])
    
    
    model_input = Model(inputs, flat)
    
    
    shape_before_flatten = x0.shape.as_list()[1:] # [1:] to skip None
    shape_flatten = np.prod(shape_before_flatten) # value of shape in the non-batch dimension
    
#     print shape_flatten
    
    return model_input,inputs, flat,shape_flatten

def input_model_fc_deep_prebn(hidden_dim,input_shape,depth):
    inputs = Input(input_shape, name="input") 
    x = BatchNormalization()(inputs)
    x = Dense(hidden_dim, activation='linear')(x) 
    x = BatchNormalization()(x)
    x0 = swish(x)
    flat = Dropout(1./(depth+1))(x0)
    for i in range(depth):
        x = Dense(hidden_dim, activation='linear')(flat) 
        x = BatchNormalization()(x)
        x0 = swish(x)
        flat0 = Dropout(1./(depth+1))(x0)
        flat = add([flat,flat0])
    
    
    model_input = Model(inputs, flat)
    
    
    shape_before_flatten = x0.shape.as_list()[1:] # [1:] to skip None
    shape_flatten = np.prod(shape_before_flatten) # value of shape in the non-batch dimension
    
#     print shape_flatten
    
    return model_input,inputs, flat,shape_flatten


# 
def input_model_fc_deep_lastdrop(hidden_dim,input_shape,depth):
    inputs = Input(input_shape, name="input") 
    x = Dense(hidden_dim, activation='linear')(inputs) 
    x = BatchNormalization()(x)
    x0 = swish(x)
    flat =  x0
    for i in range(depth):
        x = Dense(hidden_dim, activation='linear')(flat) 
        x = BatchNormalization()(x)
        x0 = swish(x)
        flat0 =  x0
        flat = add([flat,flat0])
    
    flat = Dropout(0.5)(flat)
    
    model_input = Model(inputs, flat)
    
    
    shape_before_flatten = x0.shape.as_list()[1:] # [1:] to skip None
    shape_flatten = np.prod(shape_before_flatten) # value of shape in the non-batch dimension
    
#     print shape_flatten
    
    return model_input,inputs, flat,shape_flatten

def embedding_model(class_num,feature_dim):
    input_target = Input(shape=(1,))
    center = Embedding(class_num,feature_dim)(input_target)    
    model = Model(input_target,center)
    opt = Adam(lr=1e-3,beta_1=0.5,beta_2=0.999) #SGD(lr=lr)   
    model.compile(loss='mse', optimizer=opt) 
    return model

    

 