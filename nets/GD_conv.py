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
 
def get_generative(input_shape=None, dense_dim=200, out_dim=50, lr=1e-3):
    G_in = Input(input_shape)
    x = _residual_block(basic_block, filters=out_dim, repetitions=2)(G_in)
    G_out = _bn_relu(x)
     
    shape_before_flatten = G_out.shape.as_list()[1:] # [1:] to skip None
    G = Model(G_in, G_out)
    opt = Adam(lr=lr,beta_1=0.5,beta_2=0.999) #SGD(lr=lr)   
    G.compile(loss='mse', optimizer=opt)   
    return G, G_out,shape_before_flatten
 
def get_discriminative(input_shape=None,filters = 64, out_dim=50,activation='sigmoid', lr=1e-3):
    D_in = Input(input_shape)
    x = _residual_block(basic_block, filters=out_dim, repetitions=2)(D_in)
    x = _bn_relu(x)
    
    
    # Classifier block
    block_shape = K.int_shape(x)
    pool2 = AveragePooling2D(pool_size=(block_shape[ROW_AXIS], block_shape[COL_AXIS]),
                             strides=(1, 1))(x)
    flatten1 = Flatten()(pool2)
    D_out = Dense(units=out_dim, kernel_initializer="he_normal",
                  activation=activation)(flatten1)
         
    D = Model(D_in, D_out)
    dopt = Adam(lr=lr,beta_1=0.5,beta_2=0.999)
    D.compile(loss='mse', optimizer=dopt)   
    return D, D_out 