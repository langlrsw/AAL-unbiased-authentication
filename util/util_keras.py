# -*- coding: utf-8 -*-

import numpy as np
from keras import backend as K
from util import *

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def set_trainability(model, trainable=False):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable
    return

def my_get_shape(x):
    shape_before_flatten = x.shape.as_list()[1:] # [1:] to skip None
    shape_flatten = np.prod(shape_before_flatten) # value of shape in the non-batch dimension
    return shape_flatten
    
def outer_product(inputs):
    """
    inputs: list of two tensors (of equal dimensions, 
        for which you need to compute the outer product
    """
    x, y = inputs
    batchSize = K.shape(x)[0]
    outerProduct = K.expand_dims(x,axis=2) * K.expand_dims(y,axis=1)
    outerProduct = K.reshape(outerProduct, (batchSize, -1))
    # returns a flattened batch-wise set of tensors
    return outerProduct

def merge_output_multimodel(inputs_list):
    """
    inputs: list of two tensors (of equal dimensions, 
        for which you need to compute the outer product
    """
    MODEL_NUM = len(inputs_list) - 1
    y = inputs_list[-1]
    
    batchSize = K.shape(y)[0]
#     for i in range(MODEL_NUM):
#         inputs_list[i] = K.expand_dims(inputs_list[i],axis=1)
    x = K.concatenate([K.expand_dims(inputs_list[i],axis=1) for i in range(MODEL_NUM)],axis=1)
    
    outerProduct = K.sum(x * K.expand_dims(y,axis=2),axis=1)
#     outerProduct = K.reshape(outerProduct, (batchSize, -1))
    # returns a flattened batch-wise set of tensors
    return outerProduct

def prob2extreme(prob):
    scaled_prob = 2 * prob - 1
    extremed_prob=np.sin(scaled_prob*np.pi/2)
    return (extremed_prob+1)/2



from collections import OrderedDict

def param2numpy(model_tran,mean_x=None,std_x=None):
    
    param_dic = OrderedDict()
    if mean_x is not None:
        param_dic['mean_x'] = mean_x
        param_dic['std_x'] = std_x
    for layer in model_tran.layers:
        if layer.name[:5] == 'input':
            continue

        if layer.name == 'model_1':
            depth = len(layer.layers)/6
            p_drop = 1./depth

        for sub_layer in layer.layers:
            if sub_layer.name[:5] == 'input':
                continue
            elif sub_layer.name[:5] == 'dense' or sub_layer.name[:5] == 'batch':
                param_dic[sub_layer.name] = sub_layer.get_weights()
            elif sub_layer.name[:4] == 'drop':
                param_dic[sub_layer.name] = p_drop
            else:
                param_dic[sub_layer.name] = 0
    return param_dic

def param2numpy_transfer(model_tran,mean_x=None,std_x=None):
    
    param_dic = OrderedDict()
    if mean_x is not None:
        param_dic['mean_x'] = mean_x
        param_dic['std_x'] = std_x
    flag_break = False
    for layer in model_tran.layers:
        if flag_break:
            break
        if layer.name[:5] == 'input':
            continue
        if layer.name == 'model_2':
            flag_break = True
            

        if layer.name == 'model_1':
            depth = len(layer.layers)/6
            p_drop = 1./depth

        for sub_layer in layer.layers:
            if sub_layer.name[:5] == 'input':
                continue
            elif sub_layer.name[:5] == 'dense' or sub_layer.name[:5] == 'batch':
                param_dic[sub_layer.name] = sub_layer.get_weights()
            elif sub_layer.name[:4] == 'drop':
                param_dic[sub_layer.name] = p_drop
            else:
                param_dic[sub_layer.name] = 0
    return param_dic



def my_dense(x,W,b):
    return np.dot(x,W)+b[np.newaxis,:]
def my_sigmoid(x,alpha=1):
    return 1./(1.+np.exp(-alpha*x))
def my_swish(x):
    return x*my_sigmoid(x)
def my_undropout(x,p=0.5):
    return x*p
def my_batchnorm(x,gamma,beta,moving_mean,moving_variance):
    inv = 1./np.sqrt(moving_variance + 1e-3)
    inv = inv * gamma
    return x * inv + beta - moving_mean * inv
def my_tanh(x):
    return my_sigmoid(x,alpha=2) * 2. - 1.

def my_zscore(X):
    mean_X = np.mean(X, 0)
    std_X = np.std(X, 0)
    std_X[np.where(std_X == 0.0)] = 1.0
    return (X - mean_X) / std_X, mean_X, std_X


def my_zscore_test(X, mean_X, std_X):
    std_X[np.where(std_X == 0.0)] = 1.0
    return (X - mean_X) / std_X

def transformer(x,param_dic,flag_dropout=False):
    x = x.copy()
#     print param_dic.keys()
    if param_dic.get('mean_x',None) is not None:
        mean_x = param_dic['mean_x']
        std_x = param_dic['std_x']
        x = my_zscore_test(x,mean_x,std_x)
    
    n = len(param_dic)
    flag_start = True
    for i_layer,layer_name in enumerate(param_dic):
        if layer_name[:5] == 'dense':
            if flag_start:
                flag_start = False
            else:
                pre_x = x.copy()
            param = param_dic[layer_name]
            x = my_dense(x,param[0],param[1])
#             if i_layer == n - 1:
                      
        elif layer_name[:5] == 'batch':
            param = param_dic[layer_name]
            x = my_batchnorm(x,param[0],param[1],param[2],param[3])
        elif layer_name[:10] == 'activation':
            if i_layer == n - 1:
                x=my_tanh(x)
            else:
                x=my_swish(x)
        elif layer_name[:3] == 'add':
            x = x + pre_x
        elif layer_name[:4] == 'drop':
            if flag_dropout:
                param = param_dic[layer_name]
#                 x = my_undropout(x,p=param)       
    
    return x

def transformer2prob(x,param_dic,flag_dropout=False):
    x = x.copy()
    n = len(param_dic)
    flag_start = True
    for i_layer,layer_name in enumerate(param_dic):
        if layer_name[:5] == 'dense':
            if flag_start:
                flag_start = False
            else:
                pre_x = x.copy()
            param = param_dic[layer_name]
            x = my_dense(x,param[0],param[1])
            if i_layer == n - 2:
                 x = my_tanh(x)     
        elif layer_name[:5] == 'batch':
            param = param_dic[layer_name]
            x = my_batchnorm(x,param[0],param[1],param[2],param[3])
        elif layer_name[:10] == 'activation':
            x = my_swish(x)
        elif layer_name[:3] == 'add':
            x = x + pre_x
        elif layer_name[:4] == 'drop':
            if flag_dropout:
                param = param_dic[layer_name]
#                 x = my_undropout(x,p=param)                  
    return x


import pickle

def save_param(folder_path,file_name,param_dic):
    mkdir(folder_path)
    pickle.dump( param_dic, open( os.path.join(folder_path,file_name), "wb" ) )

def load_param(folder_path,file_name):
    param_dic = pickle.load( open( os.path.join(folder_path,file_name), "rb" ) )
    return param_dic

def np_param_predict(x,model_tran,flag_dropout=False):
    param_dic = param2numpy(model_tran)
    return transformer(x,param_dic,flag_dropout=flag_dropout)

def np_param_predict2prob(x,model_tran,flag_dropout=False):
    param_dic = param2numpy(model_tran)
    return transformer2prob(x,param_dic,flag_dropout=flag_dropout)

def scheduler(model,lr):
    K.set_value(model.optimizer.lr, lr)
    return K.get_value(model.optimizer.lr)

def scheduler_fixdecay(model,ratio):
    lr = K.get_value(model.optimizer.lr)
    K.set_value(model.optimizer.lr, lr*ratio)
    return K.get_value(model.optimizer.lr)

def layer_select_col(x,feature_dim=128):
    return x[:,:feature_dim]
 