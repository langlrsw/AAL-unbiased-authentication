# -*- coding: utf-8 -*-

import numpy as np
import h5py
import os.path
from sklearn.metrics import roc_auc_score
 

from keras.models import Sequential, Model
from keras.layers import Convolution2D, MaxPooling2D, Dense, Dropout, Flatten, InputLayer, Input, merge,concatenate,add,Lambda,multiply
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

from keras import regularizers
from keras import initializers

from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
import tensorflow as tf
from keras.engine import *

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

class NegDistLayer(Layer):

    def __init__(self, output_dim, **kwargs):
#         print 'output_dim',output_dim
        self.output_dim = output_dim
        super(NegDistLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(NegDistLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):

        distmat1 = K.repeat_elements(K.sum(K.pow(x, 2),axis=1, keepdims=True),self.output_dim,axis=1) 
        distmat2 = K.expand_dims(K.mean(K.ones_like(x),axis=1),axis=1) * K.expand_dims(K.sum(K.pow(self.kernel, 2),axis=0),axis=0)
        
#         print 'distmat1.get_shape(),distmat2.get_shape()',distmat1.get_shape(),distmat2.get_shape()
#                    K.repeat_elements(K.expand_dims(K.sum(K.pow(self.kernel, 2),axis=0),axis=0),1024,axis=0)
        distmat = distmat1 + distmat2 - 2. * K.dot(x, self.kernel)
#         return -K.sqrt(K.clip(distmat,1e-12,None))
        return -distmat * 0.5
#         return -K.log(K.sqrt(K.clip(distmat,1e-12,None)))
#         return -K.log(K.clip(distmat,1e-12,None))

#         return -K.sqrt(K.clip(distmat/self.output_dim,1e-12,None))
#         return -distmat/self.output_dim
        
     
#         return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
def swish(x):
    y = Activation('sigmoid')(x)
    return multiply([x,y])
 
def ada_eye(mat):
    col = K.tf.reduce_sum(mat, 1)
    col = K.tf.ones_like(col)
    return K.tf.diag(col)

def orth_reg(W):
    WT = K.transpose(W)
    X = K.dot(WT,W)
    X = X *(1.- ada_eye(X))
    return 1e-8 * K.sum(X*X)

 
def get_generative(input_dim=32, dense_dim=200, out_dim=50,activation='tanh', lr=1e-3,activity_l1 = 0.0,flag_orth_reg=False,flag_orth_init=False,flag_SN=False,kernel_l2=0.0):
    G_in = Input([input_dim])
 
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
   
        
    x = Dense_final(out_dim, activation='linear',activity_regularizer=activity_regu,kernel_regularizer=regu,kernel_initializer=init)(G_in)  
    if activation == 'swish':
        G_out = swish(x)
    else:
        G_out = Activation(activation)(x)
    G = Model(G_in, G_out)
    opt = Adam(lr=lr,beta_1=0.5,beta_2=0.999) #SGD(lr=lr)   
    G.compile(loss='mse', optimizer=opt)   
    return G, G_out
 
def get_discriminative(input_dim=32, out_dim=50,activation='sigmoid', lr=1e-3,kernel_l1 = 0.0,kernel_l2=0.0,flag_norm1=False,flag_orth_reg=False,flag_orth_init=False):
    D_in = Input([input_dim])
    if kernel_l1 > 0.0:
        regu = regularizers.l1(kernel_l1)
    else:
        regu = regularizers.l2(kernel_l2)
    
        
    if flag_orth_reg:
        regu = orth_reg
 
        
        
        
    if flag_orth_init:
        init = initializers.Orthogonal()
    else:
        init = 'glorot_uniform'
        
    if flag_norm1:
        x = Lambda(lambda x:K.l2_normalize(x,axis=1))(D_in)   
        D_out = Dense(out_dim, activation=activation,kernel_regularizer=regu,kernel_initializer=init)(x)
    else:
        D_out = Dense(out_dim, activation=activation,kernel_regularizer=regu,kernel_initializer=init)(D_in)
    D = Model(D_in, D_out)
    dopt = Adam(lr=lr,beta_1=0.5,beta_2=0.999)
    D.compile(loss='mse', optimizer=dopt)   
    return D, D_out
 
def get_generative_2raw(input_dim=32, dense_dim=200, out_dim=50, lr=1e-3,activity_l1 = 0.0):
    G_in = Input([input_dim])
    x = Dense(dense_dim, activation='tanh',activity_regularizer=regularizers.l1(activity_l1))(G_in) 
    x = Dense(dense_dim, activation='tanh',activity_regularizer=regularizers.l1(activity_l1))(x) 
    G_out = Dense(out_dim, activation='tanh',activity_regularizer=regularizers.l1(activity_l1))(x) 
    G = Model(G_in, G_out)
    opt = Adam(lr=lr,beta_1=0.5,beta_2=0.999) #SGD(lr=lr)   
    G.compile(loss='mse', optimizer=opt)   
    return G, G_out

def get_discriminative_softMinDist(input_dim=32, out_dim=50,activation='sigmoid', lr=1e-3,kernel_l1 = 0.0,kernel_l2=0.0):
    D_in = Input([input_dim])
    if kernel_l1 > 0.0:
        regu = regularizers.l1(kernel_l1)
    else:
        regu = regularizers.l2(kernel_l2)
        
    
#     x = Lambda(lambda x:K.l2_normalize(x,axis=1), output_shape=(input_dim, ))(D_in)    
    x = NegDistLayer(out_dim)(D_in)
    D_out = Activation(activation)(x)
    D = Model(D_in, D_out)
    dopt = Adam(lr=lr,beta_1=0.5,beta_2=0.999)
    D.compile(loss='mse', optimizer=dopt)   
    return D, D_out