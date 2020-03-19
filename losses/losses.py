# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

import numpy as np
from keras import backend as K

def my_get_shape(x):
    shape_before_flatten = x.shape.as_list()[1:] # [1:] to skip None
    shape_flatten = np.prod(shape_before_flatten) # value of shape in the non-batch dimension
    return shape_flatten
 
def my_mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)

def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)
 
def categorical_crossentropy(y_true, y_pred):
    return -K.mean(K.sum(y_true*K.log(y_pred+1e-6),axis=1), axis=-1)

def my_categorical_crossentropy(y_true, y_pred):
    return K.mean(K.categorical_crossentropy(y_true, y_pred))

def prob_mean_squared_error(y_true, y_pred):
    y_pred_mean = K.mean(y_pred,axis=0)
    l = -0.5 * K.mean(K.sum(K.square(y_pred - 0.5),axis=1), axis=-1) + 2* K.sum(K.square(y_pred_mean - 0.5),axis=-1)
    return l 

def prob_mean_squared_error_g(y_true, y_pred):
    y_pred_mean = K.mean(y_pred,axis=0)
    c = my_get_shape(y_pred)
    weight = (K.random_binomial((1,),p=0.5) * 2. - 1.)
    l = 0.5 * K.mean( weight * K.sum(K.square(y_pred - 1./c),axis=1), axis=-1) + 2* K.sum(K.square(y_pred_mean - 1./c),axis=-1)
    return l  

def logit_pmse_g(y_true, y_pred):
    y_pred = K.softmax(y_pred)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    
    y_pred_mean = K.mean(y_pred,axis=0)
    c = my_get_shape(y_pred)
    weight = (K.random_binomial((1,),p=0.5) * 2. - 1.)
    l = 0.5 * K.mean( weight * K.sum(K.square(y_pred - 1./c),axis=1), axis=-1) + 2* K.sum(K.square(y_pred_mean - 1./c),axis=-1)
    return l  


def prob_categorical_crossentropy(y_true, y_pred):
    y_pred_mean = K.mean(y_pred,axis=0)
    l= -0.5 * K.mean(K.sum(y_pred*K.log(y_pred+1e-6),axis=1), axis=-1) + 2* K.sum(y_pred_mean*K.log(y_pred_mean+1e-6),axis=-1)
    return l 

def neg_pmse(y_true, y_pred):
    y_pred_mean = K.mean(y_pred,axis=0)
    l = 0.5 * K.mean(K.sum(K.square(y_pred - 0.5),axis=1), axis=-1) + 2* K.sum(K.square(y_pred_mean - 0.5),axis=-1)
    return l 
def neg_pcce(y_true, y_pred):
    y_pred_mean = K.mean(y_pred,axis=0)
    l= 0.5 * K.mean(K.sum(y_pred*K.log(y_pred+1e-6),axis=1), axis=-1) + 2* K.sum(y_pred_mean*K.log(y_pred_mean+1e-6),axis=-1)
    return l

def logit_neg_pmse(y_true, y_pred):
    y_pred = K.softmax(y_pred)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    
    y_pred_mean = K.mean(y_pred,axis=0)
    l = 0.5 * K.mean(K.sum(K.square(y_pred - 0.5),axis=1), axis=-1) + 2* K.sum(K.square(y_pred_mean - 0.5),axis=-1)
    return l 
def logit_neg_pcce(y_true, y_pred):
    y_pred = K.softmax(y_pred)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    
    y_pred_mean = K.mean(y_pred,axis=0)
    l= 0.5 * K.mean(K.sum(y_pred*K.log(y_pred+1e-6),axis=1), axis=-1) + 2* K.sum(y_pred_mean*K.log(y_pred_mean+1e-6),axis=-1)
    return l

def pcce_tanh(y_true, y_pred):
    y_pred = (y_pred + 1.)/2.
    
    l = prob_categorical_crossentropy(y_true, K.clip(y_pred, K.epsilon(), 1) ) + prob_categorical_crossentropy(y_true, K.clip(1.-y_pred, K.epsilon(), 1))
    
    return l

def gaussian_activition_loss(y_true, y_pred):
     
    d = 128
    n = 512.
    y_pred_mean = K.mean(y_pred,axis=0)
    y_pred_sigma = K.dot(K.transpose(y_pred),y_pred)/n
    mask = 1. - K.eye(d)
    l = K.mean(K.square(y_pred_sigma*mask)) + K.mean(K.square(y_pred_mean))
    
    return l



 
def logit_sigmoid_cce(y_true, y_pred):
    y_pred = K.sum(K.exp(y_pred),axis=1)
    y_pred = y_pred / (1.+y_pred)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return K.categorical_crossentropy(y_true, y_pred) 

def logit_sigmoid_mse(y_true, y_pred):
    y_pred = K.sum(K.exp(y_pred),axis=1)
    y_pred = y_pred / (1.+y_pred)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return mean_squared_error(y_true, y_pred) 

def logit_sigmoid_cce_plus_pcce(y_true, y_pred):
    y_pred_softmax = K.softmax(y_pred)
    y_pred_softmax = K.clip(y_pred_softmax, K.epsilon(), 1)
    
    y_pred = K.sum(K.exp(y_pred),axis=1)
    y_pred = y_pred / (1.+y_pred)
    y_pred = K.expand_dims(y_pred,axis=1)
    y_pred = K.concatenate([1-y_pred,y_pred])
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return K.categorical_crossentropy(y_true, y_pred) + prob_categorical_crossentropy(y_true, y_pred) + prob_categorical_crossentropy(y_true, y_pred_softmax)

def logit_sigmoid_mse_plus_pmse(y_true, y_pred):
    y_pred_softmax = K.softmax(y_pred)
    y_pred_softmax = K.clip(y_pred_softmax, K.epsilon(), 1)
    
    y_pred = K.sum(K.exp(y_pred),axis=1)
    y_pred = y_pred / (1.+y_pred)
    y_pred = K.expand_dims(y_pred,axis=1)
    y_pred = K.concatenate([1-y_pred,y_pred])
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return mean_squared_error(y_true, y_pred) + prob_mean_squared_error(y_true, y_pred) + prob_mean_squared_error(y_true, y_pred_softmax)

def logit_sigmoid_cce_plus_neg_pcce(y_true, y_pred):
    y_pred_softmax = K.softmax(y_pred)
    y_pred_softmax = K.clip(y_pred_softmax, K.epsilon(), 1)
    
    y_pred = K.sum(K.exp(y_pred),axis=1)
    y_pred = y_pred / (1.+y_pred)
    y_pred = K.expand_dims(y_pred,axis=1)
    y_pred = K.concatenate([1-y_pred,y_pred])
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return K.categorical_crossentropy(y_true, y_pred) + prob_categorical_crossentropy(y_true, y_pred) + neg_pcce(y_true, y_pred_softmax)

def logit_sigmoid_mse_plus_neg_pmse(y_true, y_pred):
    y_pred_softmax = K.softmax(y_pred)
    y_pred_softmax = K.clip(y_pred_softmax, K.epsilon(), 1)
    
    y_pred = K.sum(K.exp(y_pred),axis=1)
    y_pred = y_pred / (1.+y_pred)
    y_pred = K.expand_dims(y_pred,axis=1)
    y_pred = K.concatenate([1-y_pred,y_pred])
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return mean_squared_error(y_true, y_pred) + prob_mean_squared_error(y_true, y_pred) + neg_pmse(y_true, y_pred_softmax)

def logit_cce(y_true, y_pred):
    y_pred = K.softmax(y_pred,axis=1)
#     y_pred = K.clip(y_pred, K.epsilon(), 1)
    return K.mean(K.categorical_crossentropy(y_true, y_pred))

def inner_product_loss(y_true, y_pred):
    return K.mean(-K.sum(y_true*y_pred))

def logit_mse(y_true, y_pred):
    y_pred = K.softmax(y_pred,axis=1)
#     y_pred = K.clip(y_pred, K.epsilon(), 1)
    return mean_squared_error(y_true, y_pred) 

def logit_cce_plus_pcce(y_true, y_pred):
    y_pred = K.softmax(y_pred,axis=1)
#     y_pred = K.clip(y_pred, K.epsilon(), 1)
    return K.categorical_crossentropy(y_true, y_pred) + prob_categorical_crossentropy(y_true, y_pred)

def logit_mse_plus_pmse(y_true, y_pred):
    y_pred = K.softmax(y_pred,axis=1)
#     y_pred = K.clip(y_pred, K.epsilon(), 1)
    return mean_squared_error(y_true, y_pred) + prob_mean_squared_error(y_true, y_pred)

def cce_plus_pcce(y_true, y_pred):
    return K.categorical_crossentropy(y_true, y_pred) + prob_categorical_crossentropy(y_true, y_pred)

def mse_plus_pmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) + prob_mean_squared_error(y_true, y_pred)


def prob_binary_crossentropy(y_true, y_pred):
    y_pred_mean = K.mean(y_pred,axis=0)
    l= -K.mean(K.sum(y_pred*K.log(y_pred+1e-6),axis=1), axis=-1)-K.mean(K.sum((1-y_pred)*K.log((1-y_pred)+1e-6),axis=1), axis=-1)  #+ K.sum(y_pred_mean*K.log(y_pred_mean+1e-6),axis=-1)
    return l
 
def norm(x):
    return K.sqrt(K.sum(x*x))
def approx_entropy_pos(feats):
    batchSize = 32
    sign = 1
    s = 0.0
    cont = 0
    for i in range(batchSize):
        for j in range(i+1,batchSize):
            s = s + K.square(K.sum(feats[i]*feats[j])/(norm(feats[i])*norm(feats[j])+1e-6))
            cont = cont + 1
    s = sign*s/cont 
    return s

def approx_entropy_neg(feats):
    batchSize = 32
    
    s = 0.0
    cont = 0
    for i in range(batchSize):
        for j in range(i+1,batchSize):
            s = s +  K.sum(feats[i]*feats[j])/(norm(feats[i])*norm(feats[j])+1e-6)
            cont = cont + 1
    s = s/cont 
    return s

def nn_triplet_loss(mask,inputs):
#     inputs,targets = tensor_list
    batch_size = 1024
    margin = 64.
    
    n = batch_size
          
        
    # Compute pairwise distance, replace by the official when merged
    dist = K.repeat_elements(K.sum(K.pow(inputs, 2),axis=1, keepdims=True),n,axis=1)
    dist = dist + K.transpose(dist)
    dist = dist * 0.5 - K.dot(inputs,K.transpose(inputs))
#     dist = K.sqrt(K.clip(dist,1e-12,None))# for numerical stability

    # For each anchor, find the hardest positive and negative
#     mask = K.dot(targets ,K.transpose(targets))
    invert_mask = 1. - mask
    
#     dm = dist*mask
    
    dm = dist+(invert_mask+K.eye(n))*1e8
    dnm = dist+mask*1e8

    dist_ap, dist_an = [], []
    for i in range(n):
        dist_ap.append(K.expand_dims(K.min(dm[i]),axis=0))
        dist_an.append(K.expand_dims(K.min(dnm[i]),axis=0))

#     ipdb.set_trace()
    dist_ap = K.concatenate(dist_ap,axis=0)
    dist_an = K.concatenate(dist_an,axis=0)

    # Compute ranking hinge loss
    loss = K.mean(K.maximum(0.0*dist_ap,dist_ap-dist_an + margin))
 
    return loss

def hard_mine_triplet_loss(mask,inputs):
#     inputs,targets = tensor_list
    batch_size = 1024
    margin = 64.
    
    n = batch_size
          
        
    # Compute pairwise distance, replace by the official when merged
    dist = K.repeat_elements(K.sum(K.pow(inputs, 2),axis=1, keepdims=True),n,axis=1)
    dist = dist + K.transpose(dist)
    dist = dist * 0.5 - K.dot(inputs,K.transpose(inputs))
#     dist = K.sqrt(K.clip(dist,1e-12,None))# for numerical stability

    # For each anchor, find the hardest positive and negative
#     mask = K.dot(targets ,K.transpose(targets))
    invert_mask = 1. - mask
    
    dm = dist*mask
    
#     dm = dist+(invert_mask+K.eye(n))*1e8
    dnm = dist+mask*1e8

    dist_ap, dist_an = [], []
    for i in range(n):
        dist_ap.append(K.expand_dims(K.max(dm[i]),axis=0))
        dist_an.append(K.expand_dims(K.min(dnm[i]),axis=0))

#     ipdb.set_trace()
    dist_ap = K.concatenate(dist_ap,axis=0)
    dist_an = K.concatenate(dist_an,axis=0)

    # Compute ranking hinge loss
    loss = K.mean(K.maximum(0.0*dist_ap,dist_ap-dist_an + margin))
 
    return loss


def hard_mine_triplet_loss2(mask,inputs):
#     inputs,targets = tensor_list
    batch_size = 1024
    margin = 64.
    
    n = batch_size
          
        
    # Compute pairwise distance, replace by the official when merged
    dist = K.repeat_elements(K.sum(K.pow(inputs, 2),axis=1, keepdims=True),n,axis=1)
    dist = dist + K.transpose(dist)
    dist = dist * 0.5 - K.dot(inputs,K.transpose(inputs))
#     dist = K.sqrt(K.clip(dist,1e-12,None))# for numerical stability

    # For each anchor, find the hardest positive and negative
#     mask = K.dot(targets ,K.transpose(targets))
    
    pos_mask = mask-K.eye(n)
    neg_mask = 1. - mask
    
    R_pos_right = K.random_uniform((n,n))
    R_neg_right = K.random_uniform((n,n))
    R_pos_left = K.random_uniform((n,n))
    R_neg_left = K.random_uniform((n,n))
    
    dm_pos = dist*pos_mask   
    dm_neg = dist*neg_mask
    
#     dist_pos = K.dot(dm_pos,R_pos_right)
#     dist_neg = K.dot(dm_neg,R_neg_right)
    
    dist_pos = K.dot(R_pos_left,dm_pos)
    dist_neg = K.dot(R_neg_left,dm_neg)
    
#     norm_pos = K.dot(pos_mask,R_pos_right)
#     norm_neg = K.dot(neg_mask,R_pos_right)
    
    loss = K.maximum(0.0*dist_pos,dist_pos-dist_neg + margin)
    
#     /K.maximum(K.ones_like(norm_pos),norm_pos-norm_neg)
  

    # Compute ranking hinge loss
    loss = K.mean(loss)
 
    return loss

def contrastive_loss_soft(mask,inputs):
#     inputs,targets = tensor_list

#     l_pcce_tanh = pcce_tanh(mask, inputs)
    
    batch_size = 1024
    margin = 16.
#     margin = 2. #diff at least 1 neuron
    
    n = batch_size
          
        
    # Compute pairwise distance, replace by the official when merged
    dist = K.repeat_elements(K.sum(K.pow(inputs, 2),axis=1, keepdims=True),n,axis=1)
    dist = dist + K.transpose(dist)
    dist = dist * 0.5 - K.dot(inputs,K.transpose(inputs))
#     dist = K.sqrt(K.clip(dist,1e-12,None))# for numerical stability

    # For each anchor, find the hardest positive and negative
#     mask = K.dot(targets ,K.transpose(targets))
    invert_mask = 1. - mask
    
    dm = dist*(mask-K.eye(n))
    
#     dm = dist+(invert_mask+K.eye(n))*1e8
    dnm = dist*invert_mask

#     dist_ap, dist_an = [], []
#     for i in range(n):
#         dist_ap.append(K.expand_dims(K.max(dm[i]),axis=0))
#         dist_an.append(K.expand_dims(K.min(dnm[i]),axis=0))

# #     ipdb.set_trace()
#     dist_ap = K.concatenate(dist_ap,axis=0)
#     dist_an = K.concatenate(dist_an,axis=0)
    
    # -ln(-x/N+1)
#     beta = 4.0*inputs.get_shape()[1]
#     print 'beta',beta
#     beta = 512
    pos_dist = dm/K.sum(mask-K.eye(n))
    neg_dist_sigmoid = 1. - K.sigmoid(-dnm + margin-1)
    neg_dist = K.log(neg_dist_sigmoid)
    neg_dist = neg_dist/(K.stop_gradient(K.sum(neg_dist_sigmoid))+1)
    
    # compute loss
    loss = K.sum(neg_dist + pos_dist)

#     # Compute ranking hinge loss
#     loss = K.mean(K.maximum(0.0*dist_ap,dist_ap-dist_an + margin))
 
    return loss

def contrastive_loss_l2norm_EMentropy(mask,inputs):
#     inputs,targets = tensor_list

    l_pcce_tanh = pcce_tanh(mask, inputs)
#     print inputs.get_shape()
    inputs = K.l2_normalize(inputs,axis=1)
    
    batch_size = 1024
    margin = 16./(2.*128.)
#     margin = 2. #diff at least 1 neuron
    
    n = batch_size
    
    
          
        
    # Compute pairwise distance, replace by the official when merged
    dist = K.repeat_elements(K.sum(K.pow(inputs, 2),axis=1, keepdims=True),n,axis=1)
    dist = dist + K.transpose(dist)
    dist = dist * 0.5 - K.dot(inputs,K.transpose(inputs))
#     dist = K.sqrt(K.clip(dist,1e-12,None))# for numerical stability

    # For each anchor, find the hardest positive and negative
#     mask = K.dot(targets ,K.transpose(targets))
    invert_mask = 1. - mask
    
    dm = dist*(mask-K.eye(n))
    
#     dm = dist+(invert_mask+K.eye(n))*1e8
    dnm = dist*invert_mask

#     dist_ap, dist_an = [], []
#     for i in range(n):
#         dist_ap.append(K.expand_dims(K.max(dm[i]),axis=0))
#         dist_an.append(K.expand_dims(K.min(dnm[i]),axis=0))

# #     ipdb.set_trace()
#     dist_ap = K.concatenate(dist_ap,axis=0)
#     dist_an = K.concatenate(dist_an,axis=0)
    
    # -ln(-x/N+1)
#     beta = 4.0*inputs.get_shape()[1]
#     print 'beta',beta
#     beta = 512
    pos_dist = dm/K.sum(mask-K.eye(n))
    neg_dist = K.maximum(0.0*dnm,-dnm + margin)
    neg_dist = neg_dist/(K.stop_gradient(K.sum(K.sign(neg_dist)))+1)
    
    # compute loss
    loss = K.sum(neg_dist + pos_dist)

#     # Compute ranking hinge loss
#     loss = K.mean(K.maximum(0.0*dist_ap,dist_ap-dist_an + margin))
 
    return loss 


 

def contra_plus_unspv_contra_l2norm(mask,inputs):
    
    l1 = contrastive_loss_l2norm(mask,inputs)
    l2 = unspv_contrastive_loss_l2norm(mask,inputs)
    return l1 + 0.1 * l2


def unspv_contrastive_loss_l2norm(mask,inputs):
#     inputs,targets = tensor_list

#     l_pcce_tanh = pcce_tanh(mask, inputs)
#     print inputs.get_shape()
    inputs = K.l2_normalize(inputs,axis=1)
    
    batch_size = 1024
    margin = 16./(2.*128.)
    threshold = 8./(2.*128.)
#     margin = 2. #diff at least 1 neuron
    
    n = batch_size
 
        
    # Compute pairwise distance, replace by the official when merged
    dist = K.repeat_elements(K.sum(K.pow(inputs, 2),axis=1, keepdims=True),n,axis=1)
    dist = dist + K.transpose(dist)
    dist = dist * 0.5 - K.dot(inputs,K.transpose(inputs))
#     dist = K.sqrt(K.clip(dist,1e-12,None))# for numerical stability

    # For each anchor, find the hardest positive and negative
#     mask = K.dot(targets ,K.transpose(targets))
    mask =  K.sigmoid((threshold - dist) * 3./threshold)
    mean_mask = K.mean(mask)

    invert_mask = 1. - mask
    diag_part = K.eye(n)*K.sigmoid(3.*K.eye(n))
    
    dm = dist
    
#     dm = dist+(invert_mask+K.eye(n))*1e8
    dnm = dist

 
    pos_dist = (mask-diag_part) * dm/K.sum(mask-diag_part)
    neg_dist = (invert_mask - K.eye(n) + diag_part) * K.maximum(0.0*dnm,-dnm + margin)
    neg_dist = neg_dist/(K.stop_gradient(K.sum(K.sign(neg_dist)))+1)
    
    # compute loss
    loss = 0.5 * K.sum(neg_dist + pos_dist) + 2.0 * K.sum(mean_mask*K.log(mean_mask+1e-6))
 
    return loss 

def anti_contrastive_loss_l2norm(mask,inputs):
#     inputs,targets = tensor_list

#     l_pcce_tanh = pcce_tanh(mask, inputs)
#     print inputs.get_shape()
    inputs = K.l2_normalize(inputs,axis=1)
    
    batch_size = 1024
     
    margin = 16./(2.*128.)
#     margin = 2. #diff at least 1 neuron
    
    n = batch_size
    
    
          
        
    # Compute pairwise distance, replace by the official when merged
    dist = K.repeat_elements(K.sum(K.pow(inputs, 2),axis=1, keepdims=True),n,axis=1)
    dist = dist + K.transpose(dist)
    dist = dist * 0.5 - K.dot(inputs,K.transpose(inputs))
#     dist = K.sqrt(K.clip(dist,1e-12,None))# for numerical stability

    # For each anchor, find the hardest positive and negative
#     mask = K.dot(targets ,K.transpose(targets))
    invert_mask = 1. - mask
    
    dm = dist*(mask-K.eye(n))
    
#     dm = dist+(invert_mask+K.eye(n))*1e8
    dnm = dist*invert_mask

#     dist_ap, dist_an = [], []
#     for i in range(n):
#         dist_ap.append(K.expand_dims(K.max(dm[i]),axis=0))
#         dist_an.append(K.expand_dims(K.min(dnm[i]),axis=0))

# #     ipdb.set_trace()
#     dist_ap = K.concatenate(dist_ap,axis=0)
#     dist_an = K.concatenate(dist_an,axis=0)
    
    # -ln(-x/N+1)
#     beta = 4.0*inputs.get_shape()[1]
#     print 'beta',beta
#     beta = 512
    pos_dist = K.maximum(0.0*dm,-dm + margin)
    pos_dist = pos_dist/(K.stop_gradient(K.sum(K.sign(pos_dist)))+1)
    neg_dist = dnm/K.sum(invert_mask)
     
    # compute loss
    loss = K.sum(neg_dist + pos_dist)

#     # Compute ranking hinge loss
#     loss = K.mean(K.maximum(0.0*dist_ap,dist_ap-dist_an + margin))
 
    return loss  

def anti_contrastive_loss_l2norm2(mask,inputs):
#     inputs,targets = tensor_list

#     l_pcce_tanh = pcce_tanh(mask, inputs)
#     print inputs.get_shape()
    inputs = K.l2_normalize(inputs,axis=1)
    
    batch_size = 512
    margin = 16./(2.*128.)
#     margin = 2. #diff at least 1 neuron
    
    n = batch_size
    
    
          
        
    # Compute pairwise distance, replace by the official when merged
    dist = K.repeat_elements(K.sum(K.pow(inputs, 2),axis=1, keepdims=True),n,axis=1)
    dist = dist + K.transpose(dist)
    dist = dist * 0.5 - K.dot(inputs,K.transpose(inputs))
#     dist = K.sqrt(K.clip(dist,1e-12,None))# for numerical stability

    # For each anchor, find the hardest positive and negative
#     mask = K.dot(targets ,K.transpose(targets))
    invert_mask = 1. - mask
    
    dm = dist*(mask-K.eye(n))
    
#     dm = dist+(invert_mask+K.eye(n))*1e8
    dnm = dist*invert_mask

#     dist_ap, dist_an = [], []
#     for i in range(n):
#         dist_ap.append(K.expand_dims(K.max(dm[i]),axis=0))
#         dist_an.append(K.expand_dims(K.min(dnm[i]),axis=0))

# #     ipdb.set_trace()
#     dist_ap = K.concatenate(dist_ap,axis=0)
#     dist_an = K.concatenate(dist_an,axis=0)
    
    # -ln(-x/N+1)
#     beta = 4.0*inputs.get_shape()[1]
#     print 'beta',beta
#     beta = 512
    pos_dist = K.maximum(0.0*dm,-dm + margin)
    pos_dist = pos_dist/(K.stop_gradient(K.sum(K.sign(pos_dist)))+1)
    neg_dist = dnm/K.sum(invert_mask)
     
    # compute loss
    loss = K.sum(neg_dist + pos_dist)

#     # Compute ranking hinge loss
#     loss = K.mean(K.maximum(0.0*dist_ap,dist_ap-dist_an + margin))
 
    return loss  

def binomial_loss_l2norm(mask,inputs):
#     inputs,targets = tensor_list

#     l_pcce_tanh = pcce_tanh(mask, inputs)
#     print inputs.get_shape()
    inputs = K.l2_normalize(inputs,axis=1)
    
    batch_size = 1024
    margin = 16./(2.*128.)
#     margin = 2. #diff at least 1 neuron
    
    n = batch_size
    
    
    alpha = 2.
    beta = 0.5
    eta = mask + 25.*(1-mask)
        
    # Compute pairwise distance, replace by the official when merged
    sim_mat = K.dot(inputs,K.transpose(inputs))
    score = -(2.*mask-1.)*alpha*(sim_mat-beta)*eta
    neg_dist = K.maximum(0.0*score,score)
    loss = neg_dist/(K.stop_gradient(K.sum(K.sign(neg_dist)))+1)
    
    loss = loss * (1.-K.eye(n))
    loss = K.mean(loss)
 
    return loss 

def contrastive_loss_l2norm(mask,inputs):
#     inputs,targets = tensor_list

#     l_pcce_tanh = pcce_tanh(mask, inputs)
#     print inputs.get_shape()
    inputs = K.l2_normalize(inputs,axis=1)
    
    batch_size = 1024
    margin = 16./(2.*128.)
#     margin = 2. #diff at least 1 neuron
    
    n = batch_size
    
    
          
        
    # Compute pairwise distance, replace by the official when merged
    dist = K.repeat_elements(K.sum(K.pow(inputs, 2),axis=1, keepdims=True),n,axis=1)
    dist = dist + K.transpose(dist)
    dist = dist * 0.5 - K.dot(inputs,K.transpose(inputs))
#     dist = K.sqrt(K.clip(dist,1e-12,None))# for numerical stability

    # For each anchor, find the hardest positive and negative
#     mask = K.dot(targets ,K.transpose(targets))
    invert_mask = 1. - mask
    
    dm = dist*(mask-K.eye(n))
    
#     dm = dist+(invert_mask+K.eye(n))*1e8
    dnm = dist*invert_mask

#     dist_ap, dist_an = [], []
#     for i in range(n):
#         dist_ap.append(K.expand_dims(K.max(dm[i]),axis=0))
#         dist_an.append(K.expand_dims(K.min(dnm[i]),axis=0))

# #     ipdb.set_trace()
#     dist_ap = K.concatenate(dist_ap,axis=0)
#     dist_an = K.concatenate(dist_an,axis=0)
    
    # -ln(-x/N+1)
#     beta = 4.0*inputs.get_shape()[1]
#     print 'beta',beta
#     beta = 512
    pos_dist = dm/K.sum(mask-K.eye(n))
    neg_dist = K.maximum(0.0*dnm,-dnm + margin)
    neg_dist = neg_dist/(K.stop_gradient(K.sum(K.sign(neg_dist)))+1)
    
    # compute loss
    loss = K.sum(neg_dist + pos_dist)

#     # Compute ranking hinge loss
#     loss = K.mean(K.maximum(0.0*dist_ap,dist_ap-dist_an + margin))
 
    return loss 



def npair8cont_mc_loss_l2norm(input_labels, inputs):
    npair_loss = npair_mc_loss_l2norm(input_labels, inputs)
    cont_loss = contrastive_loss_l2norm_direct_label(input_labels,inputs)
    return npair_loss + cont_loss

def cont8angular_loss_l2norm(input_labels,inputs):
    
    cont_loss = contrastive_loss_l2norm_direct_label(input_labels,inputs)
    angular_loss = angular_mc_loss_l2norm(input_labels, inputs)
    return 0.333*angular_loss + 0.667*cont_loss

def anti_contrastive_loss_l2norm_direct_label(input_labels,inputs):

    # do softmax cross-entropy
    lshape = K.tf.shape(input_labels)
    #assert lshape.shape == 1
    labels = K.tf.reshape(input_labels, [lshape[0], 1])

    mask = K.tf.to_float(K.tf.equal(labels, K.tf.transpose(labels)))
    
    inputs = K.l2_normalize(inputs,axis=1)
    
    batch_size = 1024
     
    margin = 16./(2.*128.)
#     margin = 2. #diff at least 1 neuron
    
    n = batch_size
    
    
          
        
    # Compute pairwise distance, replace by the official when merged
    dist = K.repeat_elements(K.sum(K.pow(inputs, 2),axis=1, keepdims=True),n,axis=1)
    dist = dist + K.transpose(dist)
    dist = dist * 0.5 - K.dot(inputs,K.transpose(inputs))
#     dist = K.sqrt(K.clip(dist,1e-12,None))# for numerical stability

    # For each anchor, find the hardest positive and negative
#     mask = K.dot(targets ,K.transpose(targets))
    invert_mask = 1. - mask
    
    dm = dist*(mask-K.eye(n))
    
#     dm = dist+(invert_mask+K.eye(n))*1e8
    dnm = dist*invert_mask

#     dist_ap, dist_an = [], []
#     for i in range(n):
#         dist_ap.append(K.expand_dims(K.max(dm[i]),axis=0))
#         dist_an.append(K.expand_dims(K.min(dnm[i]),axis=0))

# #     ipdb.set_trace()
#     dist_ap = K.concatenate(dist_ap,axis=0)
#     dist_an = K.concatenate(dist_an,axis=0)
    
    # -ln(-x/N+1)
#     beta = 4.0*inputs.get_shape()[1]
#     print 'beta',beta
#     beta = 512
    pos_dist = K.maximum(0.0*dm,-dm + margin)
    pos_dist = pos_dist/(K.stop_gradient(K.sum(K.sign(pos_dist)))+1)
    neg_dist = dnm/K.sum(invert_mask)
     
    # compute loss
    loss = K.sum(neg_dist + pos_dist)

#     # Compute ranking hinge loss
#     loss = K.mean(K.maximum(0.0*dist_ap,dist_ap-dist_an + margin))
 
    return loss  

def ada_eye(mat):
    col = K.tf.reduce_sum(mat, 1)
    col = K.tf.ones_like(col)
    return K.tf.diag(col)

def contrastive_center_loss_l2norm_direct_label(input_labels,inputs):
 
    # do softmax cross-entropy
    lshape = K.tf.shape(input_labels)
    #assert lshape.shape == 1
    labels = K.tf.reshape(input_labels, [lshape[0], 1])

    mask = K.tf.to_float(K.tf.equal(labels, K.tf.transpose(labels)))
    
    inputs = K.l2_normalize(inputs,axis=1)
    
    batch_size = 1024
    margin = 16./(2.*128.)
#     margin = 2. #diff at least 1 neuron
    
    n = batch_size
    
    
          
        
    # Compute pairwise distance, replace by the official when merged
    dist = K.repeat_elements(K.sum(K.pow(inputs, 2),axis=1, keepdims=True),n,axis=1)
    dist = dist + K.transpose(dist)
    dist = dist * 0.5 - K.dot(inputs,K.transpose(inputs))
#     dist = K.sqrt(K.clip(dist,1e-12,None))# for numerical stability

    # For each anchor, find the hardest positive and negative
#     mask = K.dot(targets ,K.transpose(targets))
    invert_mask = 1. - mask
    
    dm = dist*(mask-ada_eye(mask))
#     dm = dist*(mask-K.eye(n))
    
#     dm = dist+(invert_mask+K.eye(n))*1e8
    dnm = dist*invert_mask

#     dist_ap, dist_an = [], []
#     for i in range(n):
#         dist_ap.append(K.expand_dims(K.max(dm[i]),axis=0))
#         dist_an.append(K.expand_dims(K.min(dnm[i]),axis=0))

# #     ipdb.set_trace()
#     dist_ap = K.concatenate(dist_ap,axis=0)
#     dist_an = K.concatenate(dist_an,axis=0)
    
    # -ln(-x/N+1)
#     beta = 4.0*inputs.get_shape()[1]
#     print 'beta',beta
#     beta = 512
    pos_dist = dm/K.sum(mask-ada_eye(mask))
#     pos_dist = dm/K.sum(mask-K.eye(n))
    neg_dist = K.maximum(0.0*dnm,-dnm + margin)
    neg_dist = neg_dist/(K.stop_gradient(K.sum(K.sign(neg_dist)))+1)
    
    # compute loss
    loss = K.sum(pos_dist)

#     # Compute ranking hinge loss
#     loss = K.mean(K.maximum(0.0*dist_ap,dist_ap-dist_an + margin))
 
    return loss 

def contrastive_loss_l2norm_direct_label(input_labels,inputs):
 
    # do softmax cross-entropy
    lshape = K.tf.shape(input_labels)
    #assert lshape.shape == 1
    labels = K.tf.reshape(input_labels, [lshape[0], 1])

    mask = K.tf.to_float(K.tf.equal(labels, K.tf.transpose(labels)))
    
    inputs = K.l2_normalize(inputs,axis=1)
    
    batch_size = 1024
    margin = 16./(2.*128.)
#     margin = 2. #diff at least 1 neuron
    
    n = batch_size
    
    
          
        
    # Compute pairwise distance, replace by the official when merged
    dist = K.repeat_elements(K.sum(K.pow(inputs, 2),axis=1, keepdims=True),n,axis=1)
    dist = dist + K.transpose(dist)
    dist = dist * 0.5 - K.dot(inputs,K.transpose(inputs))
#     dist = K.sqrt(K.clip(dist,1e-12,None))# for numerical stability

    # For each anchor, find the hardest positive and negative
#     mask = K.dot(targets ,K.transpose(targets))
    invert_mask = 1. - mask
    
    dm = dist*(mask-ada_eye(mask))
#     dm = dist*(mask-K.eye(n))
    
#     dm = dist+(invert_mask+K.eye(n))*1e8
    dnm = dist*invert_mask

#     dist_ap, dist_an = [], []
#     for i in range(n):
#         dist_ap.append(K.expand_dims(K.max(dm[i]),axis=0))
#         dist_an.append(K.expand_dims(K.min(dnm[i]),axis=0))

# #     ipdb.set_trace()
#     dist_ap = K.concatenate(dist_ap,axis=0)
#     dist_an = K.concatenate(dist_an,axis=0)
    
    # -ln(-x/N+1)
#     beta = 4.0*inputs.get_shape()[1]
#     print 'beta',beta
#     beta = 512
    pos_dist = dm/K.sum(mask-ada_eye(mask))
#     pos_dist = dm/K.sum(mask-K.eye(n))
    neg_dist = K.maximum(0.0*dnm,-dnm + margin)
    neg_dist = neg_dist/(K.stop_gradient(K.sum(K.sign(neg_dist)))+1)
    
    # compute loss
    loss = K.sum(neg_dist + pos_dist)

#     # Compute ranking hinge loss
#     loss = K.mean(K.maximum(0.0*dist_ap,dist_ap-dist_an + margin))
 
    return loss 

def contrastive_loss_mean_l2norm_direct_label(input_labels,inputs):
 
    # do softmax cross-entropy
    lshape = K.tf.shape(input_labels)
    #assert lshape.shape == 1
    labels = K.tf.reshape(input_labels, [lshape[0], 1])

    mask = K.tf.to_float(K.tf.equal(labels, K.tf.transpose(labels)))
    
    inputs = K.l2_normalize(inputs,axis=1)
    
    batch_size = 1024
    margin = 16./(2.*128.)
#     margin = 2. #diff at least 1 neuron
    
    n = batch_size
    
    
    mask_mean = mask/K.tf.reduce_sum(mask, 1, keepdims=True)
    center = K.tf.matmul(mask_mean, inputs, transpose_a=False, transpose_b=False)      
        
    # Compute pairwise distance, replace by the official when merged
    dist = K.repeat_elements(K.sum(inputs*center,axis=1, keepdims=True),n,axis=1)
    dist = dist + K.transpose(dist)
    dist = dist * 0.5 - K.dot(inputs,K.transpose(center))
#     dist = K.sqrt(K.clip(dist,1e-12,None))# for numerical stability

    # For each anchor, find the hardest positive and negative
#     mask = K.dot(targets ,K.transpose(targets))
    invert_mask = 1. - mask
    
    dm = dist*(mask-K.eye(n))
    
#     dm = dist+(invert_mask+K.eye(n))*1e8
    dnm = dist*invert_mask

#     dist_ap, dist_an = [], []
#     for i in range(n):
#         dist_ap.append(K.expand_dims(K.max(dm[i]),axis=0))
#         dist_an.append(K.expand_dims(K.min(dnm[i]),axis=0))

# #     ipdb.set_trace()
#     dist_ap = K.concatenate(dist_ap,axis=0)
#     dist_an = K.concatenate(dist_an,axis=0)
    
    # -ln(-x/N+1)
#     beta = 4.0*inputs.get_shape()[1]
#     print 'beta',beta
#     beta = 512
    pos_dist = dm/K.sum(mask-K.eye(n))
    neg_dist = K.maximum(0.0*dnm,-dnm + margin)
    neg_dist = neg_dist/(K.stop_gradient(K.sum(K.sign(neg_dist)))+1)
    
    # compute loss
    loss = K.sum(neg_dist + pos_dist)

#     # Compute ranking hinge loss
#     loss = K.mean(K.maximum(0.0*dist_ap,dist_ap-dist_an + margin))
 
    return loss 


def contrastive_loss_gcnmean_l2norm_direct_label(input_labels,inputs):
 
    # do softmax cross-entropy
    lshape = K.tf.shape(input_labels)
    #assert lshape.shape == 1
    labels = K.tf.reshape(input_labels, [lshape[0], 1])

    mask = K.tf.to_float(K.tf.equal(labels, K.tf.transpose(labels)))
    
    inputs = K.l2_normalize(inputs,axis=1)
    
    batch_size = 1024
    margin = 16./(2.*128.)
#     margin = 2. #diff at least 1 neuron
    
    n = batch_size
    
    
    mask_mean = mask/K.tf.reduce_sum(mask, 1, keepdims=True)
    degree = K.tf.reduce_sum(mask, 1)
    degree = K.tf.diag(1./degree)
    gcn_mean = mask_mean + degree
    inputs = K.tf.matmul(gcn_mean, inputs, transpose_a=False, transpose_b=False)      
        
    # Compute pairwise distance, replace by the official when merged
    dist = K.repeat_elements(K.sum(K.pow(inputs, 2),axis=1, keepdims=True),n,axis=1)
    dist = dist + K.transpose(dist)
    dist = dist * 0.5 - K.dot(inputs,K.transpose(inputs))
#     dist = K.sqrt(K.clip(dist,1e-12,None))# for numerical stability

    # For each anchor, find the hardest positive and negative
#     mask = K.dot(targets ,K.transpose(targets))
    invert_mask = 1. - mask
    
    dm = dist*(mask-K.eye(n))
    
#     dm = dist+(invert_mask+K.eye(n))*1e8
    dnm = dist*invert_mask

#     dist_ap, dist_an = [], []
#     for i in range(n):
#         dist_ap.append(K.expand_dims(K.max(dm[i]),axis=0))
#         dist_an.append(K.expand_dims(K.min(dnm[i]),axis=0))

# #     ipdb.set_trace()
#     dist_ap = K.concatenate(dist_ap,axis=0)
#     dist_an = K.concatenate(dist_an,axis=0)
    
    # -ln(-x/N+1)
#     beta = 4.0*inputs.get_shape()[1]
#     print 'beta',beta
#     beta = 512
    pos_dist = dm/K.sum(mask-K.eye(n))
    neg_dist = K.maximum(0.0*dnm,-dnm + margin)
    neg_dist = neg_dist/(K.stop_gradient(K.sum(K.sign(neg_dist)))+1)
    
    # compute loss
    loss = K.sum(neg_dist + pos_dist)

#     # Compute ranking hinge loss
#     loss = K.mean(K.maximum(0.0*dist_ap,dist_ap-dist_an + margin))
 
    return loss 



def npair_loss_plus_pcce(input_labels,inputs):
    
    batch_size = 1024
    with_l2reg=True
    
#     inputs = K.l2_normalize(inputs,axis=1)

    # do softmax cross-entropy
    lshape = K.tf.shape(input_labels)
    #assert lshape.shape == 1
    labels = K.tf.reshape(input_labels, [lshape[0], 1])

    mask = K.tf.to_float(K.tf.equal(labels, K.tf.transpose(labels)))
#     mask = mask - K.tf.eye(batch_size, dtype=K.tf.float32)

    
    sample_weight = K.sum(mask,axis=1)-1.
    sample_weight = K.maximum(0.0*sample_weight,sample_weight)
    sample_weight = K.sign(sample_weight)
    
    if with_l2reg:
        reg = K.tf.reduce_mean(K.tf.reduce_sum(K.tf.square(inputs), 1))
        l2loss = K.tf.multiply(0.25 * 0.002, reg)
    else:
        l2loss = 0.0
        
#     inputs = K.tf.nn.l2_normalize(inputs)
    
    similarity_matrix = K.tf.matmul(inputs, inputs, transpose_a=False, transpose_b=True)
#     similarity_matrix += -10000. * K.tf.eye(batch_size, dtype=K.tf.float32)
    mask /= K.tf.reduce_sum(mask, 1, keepdims=True)
    xent_loss = K.tf.nn.softmax_cross_entropy_with_logits_v2(logits=similarity_matrix, labels=mask)
#     print('xent_loss.get_shape(),sample_weight.get_shape()',xent_loss.get_shape(),sample_weight.get_shape())
    # compute loss
    xent_loss = K.sum(sample_weight*xent_loss)/K.sum(sample_weight)
    
    
    sample_weight_expand = K.expand_dims(sample_weight,axis=1)
    y_pred = K.softmax(similarity_matrix)
    y_pred_mean = K.sum(y_pred*sample_weight_expand,axis=0)/K.sum(sample_weight)
    l1 = -0.5 * K.sum(y_pred*K.log(y_pred+1e-6),axis=1)
    l1 = K.sum(sample_weight*l1)/K.sum(sample_weight)
    l2 = 2* K.sum(y_pred_mean*K.log(y_pred_mean+1e-6),axis=-1)
    pcce_loss = l1+l2
 
    return l2loss + xent_loss + pcce_loss
 

def npair_loss(input_labels,inputs):
    
    batch_size = 1024
    with_l2reg=True
    
#     inputs = K.l2_normalize(inputs,axis=1)

    # do softmax cross-entropy
    lshape = K.tf.shape(input_labels)
    #assert lshape.shape == 1
    labels = K.tf.reshape(input_labels, [lshape[0], 1])

    mask = K.tf.to_float(K.tf.equal(labels, K.tf.transpose(labels)))
#     mask = mask - K.tf.eye(batch_size, dtype=K.tf.float32)

    
    sample_weight = K.sum(mask,axis=1)-1.
    sample_weight = K.maximum(0.0*sample_weight,sample_weight)
    sample_weight = K.sign(sample_weight)
    
    if with_l2reg:
        reg = K.tf.reduce_mean(K.tf.reduce_sum(K.tf.square(inputs), 1))
        l2loss = K.tf.multiply(0.25 * 0.002, reg)
    else:
        l2loss = 0.0
        
#     inputs = K.tf.nn.l2_normalize(inputs)
    
    similarity_matrix = K.tf.matmul(inputs, inputs, transpose_a=False, transpose_b=True)
#     similarity_matrix += -10000. * K.tf.eye(batch_size, dtype=K.tf.float32)
    mask /= K.tf.reduce_sum(mask, 1, keepdims=True)
    xent_loss = K.tf.nn.softmax_cross_entropy_with_logits_v2(logits=similarity_matrix, labels=mask)
#     print('xent_loss.get_shape(),sample_weight.get_shape()',xent_loss.get_shape(),sample_weight.get_shape())
    # compute loss
    xent_loss = K.sum(sample_weight*xent_loss)/K.sum(sample_weight)
 
    return l2loss + xent_loss

def npair_loss2(input_labels,inputs):
    
    batch_size = 1024
    with_l2reg=False
    
    inputs = K.l2_normalize(inputs,axis=1)

    # do softmax cross-entropy
    lshape = K.tf.shape(input_labels)
    #assert lshape.shape == 1
    labels = K.tf.reshape(input_labels, [lshape[0], 1])

    mask = K.tf.to_float(K.tf.equal(labels, K.tf.transpose(labels)))
#     mask = mask - K.tf.eye(batch_size, dtype=K.tf.float32)

    
    sample_weight = K.sum(mask,axis=1)-1.
    sample_weight = K.maximum(0.0*sample_weight,sample_weight)
    sample_weight = K.sign(sample_weight)
    
    if with_l2reg:
        reg = K.tf.reduce_mean(K.tf.reduce_sum(K.tf.square(inputs), 1))
        l2loss = K.tf.multiply(0.25 * 0.002, reg)
    else:
        l2loss = 0.0
        
#     inputs = K.tf.nn.l2_normalize(inputs)
    
    similarity_matrix = K.tf.matmul(inputs, inputs, transpose_a=False, transpose_b=True)
#     similarity_matrix += -10000. * K.tf.eye(batch_size, dtype=K.tf.float32)
    mask /= K.tf.reduce_sum(mask, 1, keepdims=True)
    xent_loss = K.tf.nn.softmax_cross_entropy_with_logits_v2(logits=similarity_matrix, labels=mask)
#     print('xent_loss.get_shape(),sample_weight.get_shape()',xent_loss.get_shape(),sample_weight.get_shape())
    # compute loss
    xent_loss = K.sum(sample_weight*xent_loss)/K.sum(sample_weight)
 
    return l2loss + xent_loss

def npair_loss_mean(input_labels,inputs):
    
    batch_size = 1024
    with_l2reg=True
    
#     inputs = K.l2_normalize(inputs,axis=1)

    # do softmax cross-entropy
    lshape = K.tf.shape(input_labels)
    #assert lshape.shape == 1
    labels = K.tf.reshape(input_labels, [lshape[0], 1])

    mask = K.tf.to_float(K.tf.equal(labels, K.tf.transpose(labels)))
#     mask = mask - K.tf.eye(batch_size, dtype=K.tf.float32)

    
    sample_weight = K.sum(mask,axis=1)-1.
    sample_weight = K.maximum(0.0*sample_weight,sample_weight)
    sample_weight = K.sign(sample_weight)
    
    if with_l2reg:
        reg = K.tf.reduce_mean(K.tf.reduce_sum(K.tf.square(inputs), 1))
        l2loss = K.tf.multiply(0.25 * 0.002, reg)
    else:
        l2loss = 0.0
        
#     inputs = K.tf.nn.l2_normalize(inputs)

    mask /= K.tf.reduce_sum(mask, 1, keepdims=True)
    center = K.tf.matmul(mask, inputs, transpose_a=False, transpose_b=False)
    similarity_matrix = K.tf.matmul(inputs, center, transpose_a=False, transpose_b=True)
#     similarity_matrix += -10000. * K.tf.eye(batch_size, dtype=K.tf.float32)
    
    xent_loss = K.tf.nn.softmax_cross_entropy_with_logits_v2(logits=similarity_matrix, labels=mask)
#     print('xent_loss.get_shape(),sample_weight.get_shape()',xent_loss.get_shape(),sample_weight.get_shape())
    # compute loss
    xent_loss = K.sum(sample_weight*xent_loss)/K.sum(sample_weight)
 
    return l2loss + xent_loss

def npair_loss_GCN_mean(input_labels,inputs):
    
    batch_size = 1024
    with_l2reg=True
    
#     inputs = K.l2_normalize(inputs,axis=1)

    # do softmax cross-entropy
    lshape = K.tf.shape(input_labels)
    #assert lshape.shape == 1
    labels = K.tf.reshape(input_labels, [lshape[0], 1])

    mask = K.tf.to_float(K.tf.equal(labels, K.tf.transpose(labels)))
#     mask = mask - K.tf.eye(batch_size, dtype=K.tf.float32)

    
    sample_weight = K.sum(mask,axis=1)-1.
    sample_weight = K.maximum(0.0*sample_weight,sample_weight)
    sample_weight = K.sign(sample_weight)
    
    if with_l2reg:
        reg = K.tf.reduce_mean(K.tf.reduce_sum(K.tf.square(inputs), 1))
        l2loss = K.tf.multiply(0.25 * 0.002, reg)
    else:
        l2loss = 0.0
        
#     inputs = K.tf.nn.l2_normalize(inputs)

    mask_mean = mask/K.tf.reduce_sum(mask, 1, keepdims=True)
    degree = K.tf.reduce_sum(mask, 1)
    degree = K.tf.diag(1./degree)
    gcn_mean = mask_mean + degree
    inputs = K.tf.matmul(gcn_mean, inputs, transpose_a=False, transpose_b=False)
    similarity_matrix = K.tf.matmul(inputs, inputs, transpose_a=False, transpose_b=True)
#     similarity_matrix += -10000. * K.tf.eye(batch_size, dtype=K.tf.float32)
    
    xent_loss = K.tf.nn.softmax_cross_entropy_with_logits_v2(logits=similarity_matrix, labels=mask_mean)
#     print('xent_loss.get_shape(),sample_weight.get_shape()',xent_loss.get_shape(),sample_weight.get_shape())
    # compute loss
    xent_loss = K.sum(sample_weight*xent_loss)/K.sum(sample_weight)
 
    return l2loss + xent_loss

def anti_npair_loss_l2norm(mask,inputs):
    
    inputs = K.l2_normalize(inputs,axis=1)

    with_l2reg=False
    
    if with_l2reg:
        reg = K.tf.reduce_mean(K.tf.reduce_sum(K.tf.square(inputs), 1))
        l2loss = K.tf.multiply(0.25 * 0.002, reg)
    else:
        l2loss = 0.0
        
#     inputs = K.tf.nn.l2_normalize(inputs)
    
    mask = 1.-mask
    
    similarity_matrix = K.tf.matmul(inputs, inputs, transpose_a=False, transpose_b=True)
    mask /= K.tf.reduce_sum(mask, 1, keepdims=True)
    xent_loss = K.tf.nn.softmax_cross_entropy_with_logits_v2(logits=similarity_matrix, labels=mask)
     
    # compute loss
    xent_loss = K.tf.reduce_mean(xent_loss)
 
    return l2loss + xent_loss
 


def contrastive_loss(mask,inputs):
#     inputs,targets = tensor_list

#     l_pcce_tanh = pcce_tanh(mask, inputs)
    
    batch_size = 1024
    margin = 16.
#     margin = 2. #diff at least 1 neuron
    
    n = batch_size
          
        
    # Compute pairwise distance, replace by the official when merged
    dist = K.repeat_elements(K.sum(K.pow(inputs, 2),axis=1, keepdims=True),n,axis=1)
    dist = dist + K.transpose(dist)
    dist = dist * 0.5 - K.dot(inputs,K.transpose(inputs))
#     dist = K.sqrt(K.clip(dist,1e-12,None))# for numerical stability

    # For each anchor, find the hardest positive and negative
#     mask = K.dot(targets ,K.transpose(targets))
    invert_mask = 1. - mask
    
    dm = dist*(mask-K.eye(n))
    
#     dm = dist+(invert_mask+K.eye(n))*1e8
    dnm = dist*invert_mask

#     dist_ap, dist_an = [], []
#     for i in range(n):
#         dist_ap.append(K.expand_dims(K.max(dm[i]),axis=0))
#         dist_an.append(K.expand_dims(K.min(dnm[i]),axis=0))

# #     ipdb.set_trace()
#     dist_ap = K.concatenate(dist_ap,axis=0)
#     dist_an = K.concatenate(dist_an,axis=0)
    
    # -ln(-x/N+1)
#     beta = 4.0*inputs.get_shape()[1]
#     print 'beta',beta
#     beta = 512
    pos_dist = dm/K.sum(mask-K.eye(n))
    neg_dist = K.maximum(0.0*dnm,-dnm + margin)
    neg_dist = neg_dist/(K.stop_gradient(K.sum(K.sign(neg_dist)))+1)
    
    # compute loss
    loss = K.sum(neg_dist + pos_dist)

#     # Compute ranking hinge loss
#     loss = K.mean(K.maximum(0.0*dist_ap,dist_ap-dist_an + margin))
 
    return loss  
  

def my_kullback_leibler_divergence(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return K.sum(y_true * K.log(y_true / y_pred), axis=-1)

def my_balanced_kullback_leibler_divergence(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return 0.5 * K.sum(y_true * K.log(y_true / y_pred), axis=-1) + 0.5 * K.sum(y_pred * K.log(y_pred / y_true), axis=-1)

def my_js_divergence(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    m = 0.5 * y_true + 0.5 * y_pred
    return 0.5 * K.sum(y_true * K.log(y_true / m), axis=1) + 0.5 * K.sum(y_pred * K.log(y_pred / m), axis=1)

def logit_focal_loss(y_true, y_pred):
    y_pred = K.softmax(y_pred,axis=1)
#     y_pred = K.clip(y_pred, K.epsilon(), 1)
    return focal_loss(y_true, y_pred) 

def focal_loss(y_true, y_pred):
    n_class = my_get_shape(y_pred) 
    gamma=2.
    alpha=1./n_class
 
    
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    loss = - (1. - y_pred)**gamma * y_true * K.log(y_pred) - alpha * y_pred**gamma * (1.-y_true) * K.log(1.-y_pred)  
    return K.sum(loss,axis=-1)

def normalize_vector(x):
     
    return K.l2_normalize(x)


def kld_(p, q):
     
    return my_kullback_leibler_divergence(p, q)

def tanh_kld_(p, q):
    
    p = (p+1.)/2.
    q = (q+1.)/2.
     
    return my_kullback_leibler_divergence(p, q)

def logit_kld_(p, q):
    p = K.softmax(p,axis=1)
    q = K.softmax(q,axis=1)
     
    return my_kullback_leibler_divergence(p, q)

def kld2_(p, q):
     
    return mean_squared_error(p, q)





def npair8angular_mc_loss_l2norm(input_labels, inputs):
    npair_loss1 = npair_loss(input_labels, inputs)
    angular_loss = angular_mc_loss_l2norm(input_labels, inputs)
    return npair_loss1 + 2.*angular_loss

def npair8angularm_loss_l2norm(input_labels, inputs):
    npair_loss1 = npair_loss(input_labels, inputs)
    angular_loss = angular_mean_loss_l2norm(input_labels, inputs)
    return npair_loss1 + 2.*angular_loss

def npair8angularm8center_loss_l2norm(input_labels, inputs):
    na_loss = npair8angularm_loss_l2norm(input_labels, inputs)
    center_loss = contrastive_center_loss_l2norm_direct_label(input_labels, inputs)
    return na_loss + 0.2 * center_loss

def npair8angularm8cont_loss_l2norm(input_labels, inputs):
    npair_loss1 = npair_loss_plus_pcce(input_labels, inputs)
    angular_loss = angular_mean_loss_l2norm(input_labels, inputs)
    cont_loss = contrastive_loss_l2norm_direct_label(input_labels, inputs)
    loss = npair_loss1 + angular_loss + cont_loss
    return loss / 3.



def npair_mc_loss_l2norm(input_labels, inputs):
    
#     inputs = K.l2_normalize(inputs,axis=1)
     
    anchor_features = inputs[1:,:]
    pos_features = inputs[:-1,:]
    sample_weight = K.tf.to_float(K.tf.equal(input_labels[1:,0], input_labels[:-1,0]))
    input_labels = input_labels[1:,:]
    
    degree=45
    batch_size=1024-1
    with_l2reg=True
    
    if with_l2reg:
        reg_anchor = K.tf.reduce_mean(K.tf.reduce_sum(K.tf.square(anchor_features), 1))
        reg_positive = K.tf.reduce_mean(K.tf.reduce_sum(K.tf.square(pos_features), 1))
        l2loss = K.tf.multiply(0.25 * 0.002, reg_anchor + reg_positive)
    else:
        l2loss = 0.0

    alpha = np.deg2rad(degree)
    sq_tan_alpha = np.tan(alpha) ** 2
    
#     anchor_features = K.l2_normalize(anchor_features,axis=1)
#     pos_features = K.l2_normalize(pos_features,axis=1)

#     anchor_features = K.tf.nn.l2_normalize(anchor_features)
#     pos_features = K.tf.nn.l2_normalize(pos_features)
    
    # 2(1+(tan(alpha))^2 * xaTxp)
    #batch_size = 10
    xaTxp = K.tf.matmul(anchor_features, pos_features, transpose_a=False, transpose_b=True)
    sim_matrix_1 = K.tf.multiply(xaTxp, K.tf.eye(batch_size, dtype=K.tf.float32))

    # 4((tan(alpha))^2(xa + xp)Txn
    xaPxpTxn = K.tf.matmul(anchor_features, pos_features, transpose_a=False, transpose_b=True)
    sim_matrix_2 = K.tf.multiply(xaPxpTxn, K.tf.ones_like(xaPxpTxn, dtype=K.tf.float32) - K.tf.eye(batch_size, dtype=K.tf.float32))

    # similarity_matrix
    similarity_matrix = sim_matrix_1 + sim_matrix_2

    # do softmax cross-entropy
    lshape = K.tf.shape(input_labels)
    #assert lshape.shape == 1
    labels = K.tf.reshape(input_labels, [lshape[0], 1])

    labels_remapped = K.tf.to_float(K.tf.equal(labels, K.tf.transpose(labels)))
    labels_remapped /= K.tf.reduce_sum(labels_remapped, 1, keepdims=True)

    xent_loss = K.tf.nn.softmax_cross_entropy_with_logits_v2(logits=similarity_matrix, labels=labels_remapped)
#     print 'xent_loss.get_shape()',xent_loss.get_shape()
    xent_loss = K.sum(sample_weight*xent_loss)/K.sum(sample_weight)

    return l2loss + xent_loss

def anti_angular_mc_loss_l2norm(input_labels, inputs):
    return - angular_mc_loss_l2norm(input_labels, inputs)


def angular_mc_loss_l2norm(input_labels, inputs):
    
    inputs = K.l2_normalize(inputs,axis=1)
     
    anchor_features = inputs[1:,:]
    pos_features = inputs[:-1,:]
    sample_weight = K.tf.to_float(K.tf.equal(input_labels[1:,0], input_labels[:-1,0]))
    input_labels = input_labels[1:,:]
    
    degree=45
    batch_size=1024-1
    with_l2reg=False
    
    if with_l2reg:
        reg_anchor = K.tf.reduce_mean(K.tf.reduce_sum(K.tf.square(anchor_features), 1))
        reg_positive = K.tf.reduce_mean(K.tf.reduce_sum(K.tf.square(pos_features), 1))
        l2loss = K.tf.multiply(0.25 * 0.002, reg_anchor + reg_positive)
    else:
        l2loss = 0.0

    alpha = np.deg2rad(degree)
    sq_tan_alpha = np.tan(alpha) ** 2
    
#     anchor_features = K.l2_normalize(anchor_features,axis=1)
#     pos_features = K.l2_normalize(pos_features,axis=1)

#     anchor_features = K.tf.nn.l2_normalize(anchor_features)
#     pos_features = K.tf.nn.l2_normalize(pos_features)
    
    # 2(1+(tan(alpha))^2 * xaTxp)
    #batch_size = 10
    xaTxp = K.tf.matmul(anchor_features, pos_features, transpose_a=False, transpose_b=True)
    sim_matrix_1 = K.tf.multiply(2.0 * (1.0 + sq_tan_alpha) * xaTxp, K.tf.eye(batch_size, dtype=K.tf.float32))

    # 4((tan(alpha))^2(xa + xp)Txn
    xaPxpTxn = K.tf.matmul((anchor_features + pos_features), pos_features, transpose_a=False, transpose_b=True)
    sim_matrix_2 = K.tf.multiply(4.0 * sq_tan_alpha * xaPxpTxn, K.tf.ones_like(xaPxpTxn, dtype=K.tf.float32) - K.tf.eye(batch_size, dtype=K.tf.float32))

    # similarity_matrix
    similarity_matrix = sim_matrix_1 + sim_matrix_2

    # do softmax cross-entropy
    lshape = K.tf.shape(input_labels)
    #assert lshape.shape == 1
    labels = K.tf.reshape(input_labels, [lshape[0], 1])

    labels_remapped = K.tf.to_float(K.tf.equal(labels, K.tf.transpose(labels)))
    labels_remapped /= K.tf.reduce_sum(labels_remapped, 1, keepdims=True)

    xent_loss = K.tf.nn.softmax_cross_entropy_with_logits_v2(logits=similarity_matrix, labels=labels_remapped)
#     print 'xent_loss.get_shape()',xent_loss.get_shape()
    xent_loss = K.sum(sample_weight*xent_loss)/K.sum(sample_weight)

    return l2loss + xent_loss

def angular_mean_loss_l2norm(input_labels, inputs):
    
    degree=45
    batch_size=1024
    with_l2reg=False
    
    # do softmax cross-entropy
    lshape = K.tf.shape(input_labels)
    #assert lshape.shape == 1
    labels = K.tf.reshape(input_labels, [lshape[0], 1])

    labels_remapped = K.tf.to_float(K.tf.equal(labels, K.tf.transpose(labels)))
    
    sample_weight = K.sum(labels_remapped,axis=1)-1.
    sample_weight = K.maximum(0.0*sample_weight,sample_weight)
    sample_weight = K.sign(sample_weight)
    
    
    labels_remapped /= K.tf.reduce_sum(labels_remapped, 1, keepdims=True)
    
    
    
    inputs = K.l2_normalize(inputs,axis=1)
     
    anchor_features = inputs
    pos_features = K.tf.matmul(labels_remapped, inputs, transpose_a=False, transpose_b=False)
#     sample_weight = K.tf.to_float(K.tf.equal(input_labels[1:,0], input_labels[:-1,0]))
#     input_labels = input_labels[1:,:]
    
    
    
    if with_l2reg:
        reg_anchor = K.tf.reduce_mean(K.tf.reduce_sum(K.tf.square(anchor_features), 1))
        reg_positive = K.tf.reduce_mean(K.tf.reduce_sum(K.tf.square(pos_features), 1))
        l2loss = K.tf.multiply(0.25 * 0.002, reg_anchor + reg_positive)
    else:
        l2loss = 0.0

    alpha = np.deg2rad(degree)
    sq_tan_alpha = np.tan(alpha) ** 2
    
#     anchor_features = K.l2_normalize(anchor_features,axis=1)
#     pos_features = K.l2_normalize(pos_features,axis=1)

#     anchor_features = K.tf.nn.l2_normalize(anchor_features)
#     pos_features = K.tf.nn.l2_normalize(pos_features)
    
    # 2(1+(tan(alpha))^2 * xaTxp)
    #batch_size = 10
    xaTxp = K.tf.matmul(anchor_features, pos_features, transpose_a=False, transpose_b=True)
    sim_matrix_1 = K.tf.multiply(2.0 * (1.0 + sq_tan_alpha) * xaTxp, K.tf.eye(batch_size, dtype=K.tf.float32))

    # 4((tan(alpha))^2(xa + xp)Txn
    xaPxpTxn = K.tf.matmul((anchor_features + pos_features), anchor_features, transpose_a=False, transpose_b=True)
    sim_matrix_2 = K.tf.multiply(4.0 * sq_tan_alpha * xaPxpTxn, K.tf.ones_like(xaPxpTxn, dtype=K.tf.float32) - K.tf.eye(batch_size, dtype=K.tf.float32))

    # similarity_matrix
    similarity_matrix = sim_matrix_1 + sim_matrix_2

    

    xent_loss = K.tf.nn.softmax_cross_entropy_with_logits_v2(logits=similarity_matrix, labels=labels_remapped)
#     print 'xent_loss.get_shape()',xent_loss.get_shape()
    xent_loss = K.sum(sample_weight*xent_loss)/K.sum(sample_weight)

    return l2loss + xent_loss

def angular_mc_loss_l2norm_plus_pcce(input_labels, inputs):
    
    inputs = K.l2_normalize(inputs,axis=1)
     
    anchor_features = inputs[1:,:]
    pos_features = inputs[:-1,:]
    sample_weight = K.tf.to_float(K.tf.equal(input_labels[1:,0], input_labels[:-1,0]))
    input_labels = input_labels[1:,:]
    
    degree=45
    batch_size=1024-1
    with_l2reg=False
    
    if with_l2reg:
        reg_anchor = K.tf.reduce_mean(K.tf.reduce_sum(K.tf.square(anchor_features), 1))
        reg_positive = K.tf.reduce_mean(K.tf.reduce_sum(K.tf.square(pos_features), 1))
        l2loss = K.tf.multiply(0.25 * 0.002, reg_anchor + reg_positive)
    else:
        l2loss = 0.0

    alpha = np.deg2rad(degree)
    sq_tan_alpha = np.tan(alpha) ** 2
    
#     anchor_features = K.l2_normalize(anchor_features,axis=1)
#     pos_features = K.l2_normalize(pos_features,axis=1)

#     anchor_features = K.tf.nn.l2_normalize(anchor_features)
#     pos_features = K.tf.nn.l2_normalize(pos_features)
    
    # 2(1+(tan(alpha))^2 * xaTxp)
    #batch_size = 10
    xaTxp = K.tf.matmul(anchor_features, pos_features, transpose_a=False, transpose_b=True)
    sim_matrix_1 = K.tf.multiply(2.0 * (1.0 + sq_tan_alpha) * xaTxp, K.tf.eye(batch_size, dtype=K.tf.float32))

    # 4((tan(alpha))^2(xa + xp)Txn
    xaPxpTxn = K.tf.matmul((anchor_features + pos_features), pos_features, transpose_a=False, transpose_b=True)
    sim_matrix_2 = K.tf.multiply(4.0 * sq_tan_alpha * xaPxpTxn, K.tf.ones_like(xaPxpTxn, dtype=K.tf.float32) - K.tf.eye(batch_size, dtype=K.tf.float32))

    # similarity_matrix
    similarity_matrix = sim_matrix_1 + sim_matrix_2

    # do softmax cross-entropy
    lshape = K.tf.shape(input_labels)
    #assert lshape.shape == 1
    labels = K.tf.reshape(input_labels, [lshape[0], 1])

    labels_remapped = K.tf.to_float(K.tf.equal(labels, K.tf.transpose(labels)))
    labels_remapped /= K.tf.reduce_sum(labels_remapped, 1, keepdims=True)

    xent_loss = K.tf.nn.softmax_cross_entropy_with_logits_v2(logits=similarity_matrix, labels=labels_remapped)
    
    sample_weight_expand = K.expand_dims(sample_weight,axis=1)
    y_pred = K.softmax(similarity_matrix)
    y_pred_mean = K.sum(y_pred*sample_weight_expand,axis=0)/K.sum(sample_weight)
    l1 = -0.5 * K.sum(y_pred*K.log(y_pred+1e-6),axis=1)
    l1 = K.sum(sample_weight*l1)/K.sum(sample_weight)
    l2 = 2* K.sum(y_pred_mean*K.log(y_pred_mean+1e-6),axis=-1)
    pcce_loss = l1+l2
    
#     print 'xent_loss.get_shape()',xent_loss.get_shape()
    xent_loss = K.sum(sample_weight*xent_loss)/K.sum(sample_weight)

    return l2loss + xent_loss + pcce_loss
  


def _get_triplet_mask(mask,n):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]
    Args:
        labels: K.int32 `Tensor` with shape [batch_size]
    """
    # Check that i, j and k are distinct
    indices_equal = K.eye(n)
    indices_not_equal = 1. - indices_equal
    i_not_equal_j = K.expand_dims(indices_not_equal, 2)
    i_not_equal_k = K.expand_dims(indices_not_equal, 1)
    j_not_equal_k = K.expand_dims(indices_not_equal, 0)

    distinct_indices = i_not_equal_j*i_not_equal_k*j_not_equal_k


    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = mask
    i_equal_j = K.expand_dims(label_equal, 2)
    i_equal_k = K.expand_dims(label_equal, 1)

    valid_labels = i_equal_j*(1.-i_equal_k)

    # Combine the two masks
    mask = distinct_indices*valid_labels

    return mask

def _get_triplet_mask_angular_loss(mask,n):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]
    Args:
        labels: K.int32 `Tensor` with shape [batch_size]
    """
    # Check that i, j and k are distinct
    indices_equal = K.eye(n)
    indices_not_equal = 1. - indices_equal
    i_not_equal_j = K.expand_dims(indices_not_equal, 2)
    i_not_equal_k = K.expand_dims(indices_not_equal, 1)
    j_not_equal_k = K.expand_dims(indices_not_equal, 0)

    distinct_indices = i_not_equal_j*i_not_equal_k*j_not_equal_k


    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = mask
    i_not_equal_j = K.expand_dims(1.-label_equal, 2)
    i_not_equal_k = K.expand_dims(1.-label_equal, 1)
    j_equal_k = K.expand_dims(label_equal, 0)

    valid_labels = i_not_equal_j*i_not_equal_k*j_equal_k

    # Combine the two masks
    mask = distinct_indices*valid_labels

    return mask




def batch_all_angular_loss_l2norm(mask, inputs):
    """Build the triplet loss over a batch of embeddings.
    We generate all the valid triplets and average the loss over the positive ones.
    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    inputs = K.l2_normalize(inputs,axis=1)
    
    batch_size = 512#1024
#     margin = 16./(2.*128.)
    degree = 36.
    rad = degree*np.pi/180.
    tan_square = np.tan(rad)**2
    
    n = batch_size
    
    #max(0,2-6*tan_square-2*(1+tan_square)*a*p+4*tan_square*n*a+4*tan_square*n*p)
      
    
    # Get the pairwise distance matrix
#     dist = K.repeat_elements(K.sum(K.pow(inputs, 2),axis=1, keepdims=True),n,axis=1)
#     dist = dist + K.transpose(dist)
#     pairwise_dist = dist * 0.5 - K.dot(inputs,K.transpose(inputs))
    
    pairwise_dist = K.dot(inputs,K.transpose(inputs))

    # shape (batch_size, batch_size, 1)
    negative_anchor_dist = K.expand_dims(pairwise_dist, 2)
 
    # shape (batch_size, 1, batch_size)
    negative_positive_dist = K.expand_dims(pairwise_dist, 1)
    
    # shape (1, batch_size, batch_size)
    anchor_positive_dist = K.expand_dims(pairwise_dist, 0)
     
        

    # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
    # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
    # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
    # and the 2nd (batch_size, 1, batch_size)
#     angular_loss = 2.-6.*tan_square-2.*(1.+tan_square)*anchor_positive_dist+4.*tan_square*negative_anchor_dist+4.*tan_square*negative_positive_dist
    angular_loss = -2.*(1.+tan_square)*anchor_positive_dist+4.*tan_square*negative_anchor_dist+4.*tan_square*negative_positive_dist

    # Put to zero the invalid triplets
    # (where label(a) != label(p) or label(n) == label(a) or a == p)
    mask = _get_triplet_mask_angular_loss(mask,n)
    triplet_loss = mask * angular_loss

    # Remove negative losses (i.e. the easy triplets)
    triplet_loss = K.maximum(triplet_loss, 0.0)

    # Count number of positive triplets (where triplet_loss > 0)
    valid_triplets = K.sign(triplet_loss)
    num_positive_triplets = K.sum(valid_triplets)
    num_valid_triplets = K.sum(mask)
#     fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1)

    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = K.sum(triplet_loss) / (num_positive_triplets + 1)

    return triplet_loss 


def anti_batch_all_triplet_loss_balance_l2norm(mask, inputs):
    """Build the triplet loss over a batch of embeddings.
    We generate all the valid triplets and average the loss over the positive ones.
    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    inputs = K.l2_normalize(inputs,axis=1)
    
    batch_size = 512#1024
    margin = 16./(2.*128.)
    
    n = batch_size
      
    
    # Get the pairwise distance matrix
    dist = K.repeat_elements(K.sum(K.pow(inputs, 2),axis=1, keepdims=True),n,axis=1)
    dist = dist + K.transpose(dist)
    pairwise_dist = dist * 0.5 - K.dot(inputs,K.transpose(inputs))

    # shape (batch_size, batch_size, 1)
    anchor_positive_dist = K.expand_dims(pairwise_dist, 2)
 
    # shape (batch_size, 1, batch_size)
    anchor_negative_dist = K.expand_dims(pairwise_dist, 1)
     

    # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
    # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
    # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
    # and the 2nd (batch_size, 1, batch_size)
    triplet_loss =  K.abs(anchor_negative_dist - anchor_positive_dist)

    # Put to zero the invalid triplets
    # (where label(a) != label(p) or label(n) == label(a) or a == p)
    mask = _get_triplet_mask(mask,n)
#     triplet_loss = mask * triplet_loss

    # Remove negative losses (i.e. the easy triplets)
#     triplet_loss = K.maximum(triplet_loss, 0.0)

    # Count number of positive triplets (where triplet_loss > 0)
    valid_triplets = K.sign(triplet_loss)
    num_positive_triplets = K.sum(valid_triplets)
    num_valid_triplets = K.sum(mask)
#     fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1)

    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = K.sum(triplet_loss) / (num_positive_triplets + 1)

    return triplet_loss 

def anti_batch_all_triplet_loss_l2norm(mask, inputs):
    """Build the triplet loss over a batch of embeddings.
    We generate all the valid triplets and average the loss over the positive ones.
    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    inputs = K.l2_normalize(inputs,axis=1)
    
    batch_size = 512#1024
    margin = 16./(2.*128.)
    
    n = batch_size
      
    
    # Get the pairwise distance matrix
    dist = K.repeat_elements(K.sum(K.pow(inputs, 2),axis=1, keepdims=True),n,axis=1)
    dist = dist + K.transpose(dist)
    pairwise_dist = dist * 0.5 - K.dot(inputs,K.transpose(inputs))

    # shape (batch_size, batch_size, 1)
    anchor_positive_dist = K.expand_dims(pairwise_dist, 2)
 
    # shape (batch_size, 1, batch_size)
    anchor_negative_dist = K.expand_dims(pairwise_dist, 1)
     

    # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
    # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
    # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
    # and the 2nd (batch_size, 1, batch_size)
    triplet_loss =  anchor_negative_dist - anchor_positive_dist + margin

    # Put to zero the invalid triplets
    # (where label(a) != label(p) or label(n) == label(a) or a == p)
    mask = _get_triplet_mask(mask,n)
#     triplet_loss = mask * triplet_loss

    # Remove negative losses (i.e. the easy triplets)
    triplet_loss = K.maximum(triplet_loss, 0.0)

    # Count number of positive triplets (where triplet_loss > 0)
    valid_triplets = K.sign(triplet_loss)
    num_positive_triplets = K.sum(valid_triplets)
    num_valid_triplets = K.sum(mask)
#     fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1)

    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = K.sum(triplet_loss) / (num_positive_triplets + 1)

    return triplet_loss 

def batch_all_triplet_loss_l2norm(mask, inputs):
    """Build the triplet loss over a batch of embeddings.
    We generate all the valid triplets and average the loss over the positive ones.
    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    inputs = K.l2_normalize(inputs,axis=1)
    
    batch_size = 512#1024
 
    margin = 16./(2.*128.)
    
    n = batch_size
      
    
    # Get the pairwise distance matrix
    dist = K.repeat_elements(K.sum(K.pow(inputs, 2),axis=1, keepdims=True),n,axis=1)
    dist = dist + K.transpose(dist)
    pairwise_dist = dist * 0.5 - K.dot(inputs,K.transpose(inputs))

    # shape (batch_size, batch_size, 1)
    anchor_positive_dist = K.expand_dims(pairwise_dist, 2)
 
    # shape (batch_size, 1, batch_size)
    anchor_negative_dist = K.expand_dims(pairwise_dist, 1)
     

    # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
    # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
    # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
    # and the 2nd (batch_size, 1, batch_size)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    # Put to zero the invalid triplets
    # (where label(a) != label(p) or label(n) == label(a) or a == p)
    mask = _get_triplet_mask(mask,n)
    triplet_loss = mask * triplet_loss

    # Remove negative losses (i.e. the easy triplets)
    triplet_loss = K.maximum(triplet_loss, 0.0)

    # Count number of positive triplets (where triplet_loss > 0)
    valid_triplets = K.sign(triplet_loss)
    num_positive_triplets = K.sum(valid_triplets)
    num_valid_triplets = K.sum(mask)
#     fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1)

    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = K.sum(triplet_loss) / (num_positive_triplets + 1)

    return triplet_loss 


def batch_all_triplet_loss(mask, inputs):
    """Build the triplet loss over a batch of embeddings.
    We generate all the valid triplets and average the loss over the positive ones.
    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    batch_size = 512
    margin = 16.
    
    n = batch_size
      
    
    # Get the pairwise distance matrix
    dist = K.repeat_elements(K.sum(K.pow(inputs, 2),axis=1, keepdims=True),n,axis=1)
    dist = dist + K.transpose(dist)
    pairwise_dist = dist * 0.5 - K.dot(inputs,K.transpose(inputs))

    # shape (batch_size, batch_size, 1)
    anchor_positive_dist = K.expand_dims(pairwise_dist, 2)
 
    # shape (batch_size, 1, batch_size)
    anchor_negative_dist = K.expand_dims(pairwise_dist, 1)
     

    # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
    # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
    # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
    # and the 2nd (batch_size, 1, batch_size)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    # Put to zero the invalid triplets
    # (where label(a) != label(p) or label(n) == label(a) or a == p)
    mask = _get_triplet_mask(mask,n)
    triplet_loss = mask * triplet_loss

    # Remove negative losses (i.e. the easy triplets)
    triplet_loss = K.maximum(triplet_loss, 0.0)

    # Count number of positive triplets (where triplet_loss > 0)
    valid_triplets = K.sign(triplet_loss)
    num_positive_triplets = K.sum(valid_triplets)
    num_valid_triplets = K.sum(mask)
#     fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1)

    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = K.sum(triplet_loss) / (num_positive_triplets + 1)

    return triplet_loss 


def anti_batch_hard_triplet_loss_l2norm(mask, inputs):
    """Build the triplet loss over a batch of embeddings.
    For each anchor, we get the hardest positive and hardest negative to form a triplet.
    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
     
    inputs = K.l2_normalize(inputs,axis=1)
    
    batch_size = 1024
    margin = 16./(2.*128.)
    
    n = batch_size
   
    # Get the pairwise distance matrix
    dist = K.repeat_elements(K.sum(K.pow(inputs, 2),axis=1, keepdims=True),n,axis=1)
    dist = dist + K.transpose(dist)
    pairwise_dist = dist * 0.5 - K.dot(inputs,K.transpose(inputs))
     

    # For each anchor, get the hardest positive
    # First, we need to get a mask for every valid positive (they should have same label)
    mask_anchor_positive = mask-K.eye(n)
    

    # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
    anchor_positive_dist = mask_anchor_positive*pairwise_dist

    # shape (batch_size, 1)
    hardest_positive_dist = K.max(anchor_positive_dist, axis=1, keepdims=True)
   

    # For each anchor, get the hardest negative
    # First, we need to get a mask for every valid negative (they should have different labels)
    mask_anchor_negative = 1. - mask
  

    # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
    max_anchor_negative_dist = K.max(pairwise_dist, axis=1, keepdims=True)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

    # shape (batch_size,)
    hardest_negative_dist = K.min(anchor_negative_dist, axis=1, keepdims=True)
 

    # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
#     triplet_loss = K.maximum(hardest_positive_dist - hardest_negative_dist + margin, 0.0*hardest_negative_dist)
    
    triplet_loss = K.abs(hardest_positive_dist - hardest_negative_dist)

    # Get final mean triplet loss
    triplet_loss = K.mean(triplet_loss)

    return triplet_loss

def batch_hard_triplet_loss_l2norm(mask, inputs):
    """Build the triplet loss over a batch of embeddings.
    For each anchor, we get the hardest positive and hardest negative to form a triplet.
    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
     
    inputs = K.l2_normalize(inputs,axis=1)
    
    batch_size = 1024
    margin = 16./(2.*128.)
    
    n = batch_size
   
    # Get the pairwise distance matrix
    dist = K.repeat_elements(K.sum(K.pow(inputs, 2),axis=1, keepdims=True),n,axis=1)
    dist = dist + K.transpose(dist)
    pairwise_dist = dist * 0.5 - K.dot(inputs,K.transpose(inputs))
     

    # For each anchor, get the hardest positive
    # First, we need to get a mask for every valid positive (they should have same label)
    mask_anchor_positive = mask-K.eye(n)
    

    # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
    anchor_positive_dist = mask_anchor_positive*pairwise_dist

    # shape (batch_size, 1)
    hardest_positive_dist = K.max(anchor_positive_dist, axis=1, keepdims=True)
   

    # For each anchor, get the hardest negative
    # First, we need to get a mask for every valid negative (they should have different labels)
    mask_anchor_negative = 1. - mask
  

    # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
    max_anchor_negative_dist = K.max(pairwise_dist, axis=1, keepdims=True)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

    # shape (batch_size,)
    hardest_negative_dist = K.min(anchor_negative_dist, axis=1, keepdims=True)
 

    # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
    triplet_loss = K.maximum(hardest_positive_dist - hardest_negative_dist + margin, 0.0*hardest_negative_dist)

    # Get final mean triplet loss
    triplet_loss = K.mean(triplet_loss)

    return triplet_loss

def batch_hard_triplet_loss(mask, inputs):
    """Build the triplet loss over a batch of embeddings.
    For each anchor, we get the hardest positive and hardest negative to form a triplet.
    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
     
    
    
    batch_size = 1024
    margin = 16.
    
    n = batch_size
          
        
     
    
    
    
    # Get the pairwise distance matrix
    dist = K.repeat_elements(K.sum(K.pow(inputs, 2),axis=1, keepdims=True),n,axis=1)
    dist = dist + K.transpose(dist)
    pairwise_dist = dist * 0.5 - K.dot(inputs,K.transpose(inputs))
     

    # For each anchor, get the hardest positive
    # First, we need to get a mask for every valid positive (they should have same label)
    mask_anchor_positive = mask-K.eye(n)
    

    # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
    anchor_positive_dist = mask_anchor_positive*pairwise_dist

    # shape (batch_size, 1)
    hardest_positive_dist = K.max(anchor_positive_dist, axis=1, keepdims=True)
   

    # For each anchor, get the hardest negative
    # First, we need to get a mask for every valid negative (they should have different labels)
    mask_anchor_negative = 1. - mask
  

    # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
    max_anchor_negative_dist = K.max(pairwise_dist, axis=1, keepdims=True)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

    # shape (batch_size,)
    hardest_negative_dist = K.min(anchor_negative_dist, axis=1, keepdims=True)
 

    # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
    triplet_loss = K.maximum(hardest_positive_dist - hardest_negative_dist + margin, 0.0*hardest_negative_dist)

    # Get final mean triplet loss
    triplet_loss = K.mean(triplet_loss)

    return triplet_loss

