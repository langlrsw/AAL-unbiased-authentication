# -*- coding: utf-8 -*-

import numpy as np
import h5py
import os.path
from sklearn.metrics import roc_auc_score
 

from keras.models import Sequential, Model
from keras.layers import *
from keras import backend as K
from keras.utils.np_utils import to_categorical
from keras.layers.advanced_activations import PReLU,ELU,LeakyReLU

from keras.models import Model
from keras.layers import Dense, Activation, Input, Reshape
from keras.layers import Conv1D, Flatten, Dropout,Conv2D
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

from resnet2 import _residual_block,basic_block,_bn_relu

from keras.layers.convolutional import (
    Conv2D,
    MaxPooling2D,
    AveragePooling2D
)

ROW_AXIS = 1
COL_AXIS = 2
CHANNEL_AXIS = 3

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

def get_generative(input_shape=None, dense_dim=200, out_dim=50, lr=1e-3):
    G_in = Input(input_shape)
    G_out = Conv2D(out_dim, kernel_size=3, strides=(1, 1), padding='same', activation='tanh', kernel_initializer="he_normal")(G_in)
    shape_before_flatten = G_out.shape.as_list()[1:] # [1:] to skip None
    G = Model(G_in, G_out)
    opt = Adam(lr=lr,beta_1=0.5,beta_2=0.999) #SGD(lr=lr)   
    G.compile(loss='mse', optimizer=opt)   
    return G, G_out,shape_before_flatten
 
def get_discriminative(input_shape=None,filters = 64, out_dim=50,activation='sigmoid', lr=1e-3,kernel_l1 = 0.0,kernel_l2=0.0):
    if kernel_l1 > 0.0:
        regu = regularizers.l1(kernel_l1)
    else:
        regu = regularizers.l2(kernel_l2)
        
    D_in = Input(input_shape)
    flatten1 = Flatten()(D_in)
    D_out = Dense(out_dim, activation=activation,kernel_regularizer=regu)(flatten1) 
    D = Model(D_in, D_out)
    dopt = Adam(lr=lr,beta_1=0.5,beta_2=0.999)
    D.compile(loss='mse', optimizer=dopt)   
    return D, D_out 
 
def get_discriminative_softMinDist(input_shape=None,filters = 64, out_dim=50,activation='sigmoid', lr=1e-3):
    D_in = Input(input_shape)
    flatten1 = Flatten()(D_in)
    x = NegDistLayer(out_dim)(flatten1) 
    D_out = Activation(activation)(x)
     
    D = Model(D_in, D_out)
    dopt = Adam(lr=lr,beta_1=0.5,beta_2=0.999)
    D.compile(loss='mse', optimizer=dopt)   
    return D, D_out 