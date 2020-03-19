# -*- coding: utf-8 -*-

import numpy as np
import h5py
import os 
from sklearn.metrics import roc_auc_score
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
import time as time_simple
from util_keras import *
from keras.models import load_model
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier


from keras.layers.core import Layer
from keras.engine import InputSpec
from keras import backend as K
from keras import initializers



def idx_in_subset(subset,y):
    fullset = np.unique(y)
    # diffset = np.setdiff1d(fullset,subset)
    dic_binary = {e:0 for e in fullset}
    for e in subset:
        dic_binary[e] = 1
    idx = np.array(map(lambda x: dic_binary[x],y))
    return idx == 1


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

def str_array_add(array_tuple):
    for i,e in enumerate(array_tuple):
        if i == 0:
            a = e
        else:
            a = np.core.defchararray.add(a,e)
    return a

def gen_identical_mask(y):
    n = len(y)
    y = y[:,np.newaxis]
    y = np.repeat(y,n,axis=1)
#     print y.shape
    return (y==y.T).astype(int)

def get_anchor_positive_triplet_mask(labels):
    """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.
    Args:
        labels: K.int32 `Tensor` with shape [batch_size]
    Returns:
        mask: K.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check that i and j are distinct
    indices_equal = K.cast(K.eye(K.shape(labels)[0]), K.bool)
    indices_not_equal = K.logical_not(indices_equal)

    # Check if labels[i] == labels[j]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = K.equal(K.expand_dims(labels, 0), K.expand_dims(labels, 1))

    # Combine the two masks
    mask = K.logical_and(indices_not_equal, labels_equal)

    return mask


def get_anchor_negative_triplet_mask(labels):
    """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.
    Args:
        labels: K.int32 `Tensor` with shape [batch_size]
    Returns:
        mask: K.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check if labels[i] != labels[k]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = K.equal(K.expand_dims(labels, 0), K.expand_dims(labels, 1))

    mask = K.logical_not(labels_equal)

    return mask


def get_triplet_mask(labels):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]
    Args:
        labels: K.int32 `Tensor` with shape [batch_size]
    """
    # Check that i, j and k are distinct
    indices_equal = K.cast(K.eye(K.shape(labels)[0]), K.bool)
    indices_not_equal = K.logical_not(indices_equal)
    i_not_equal_j = K.expand_dims(indices_not_equal, 2)
    i_not_equal_k = K.expand_dims(indices_not_equal, 1)
    j_not_equal_k = K.expand_dims(indices_not_equal, 0)

    distinct_indices = K.logical_and(K.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)


    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = K.equal(K.expand_dims(labels, 0), K.expand_dims(labels, 1))
    i_equal_j = K.expand_dims(label_equal, 2)
    i_equal_k = K.expand_dims(label_equal, 1)

    valid_labels = K.logical_and(i_equal_j, K.logical_not(i_equal_k))

    # Combine the two masks
    mask = K.logical_and(distinct_indices, valid_labels)

    return mask

 
def label2uniqueID_sub(y):
    dic ={}
    uni_set = np.unique(y)
    y_new = y.copy()
    for i,e in enumerate(uni_set):
        y_new[y==e]=i
        dic[e] = i
 
    return y_new[:,np.newaxis],dic

def label2uniqueID(Y):
    dic_list = []
    for i in xrange(Y.shape[1]):
        y,dic = label2uniqueID_sub(Y[:,i])
        if i == 0:
            new_Y = y
        else:
            new_Y = np.hstack([new_Y,y])
        dic_list.append(dic)
 
    return new_Y,dic_list

def label2uniqueID_sub_test(y,dic):
   
    y_new = y.copy()
    for e in dic:
        y_new[y==e]=dic[e]

    return y_new[:,np.newaxis] 

def label2uniqueID_test(Y,dic_list):
    
    for i in range(Y.shape[1]):
        y = label2uniqueID_sub_test(Y[:,i],dic_list[i]) 
        if i == 0:
            new_Y = y
        else:
            new_Y = np.hstack([new_Y,y])
 
    return new_Y

def label2uniqueID_sub_train(y):
    dic ={}
    uni_set = np.unique(y)
    y_new = y.copy()
    for i,e in enumerate(uni_set):
        y_new[y==e]=i
        dic[e] = i
 
    return y_new[:,np.newaxis].astype(int),dic

def label2uniqueID_train(Y):
    dic_list = []
    for i in range(Y.shape[1]):
        y,dic = label2uniqueID_sub_train(Y[:,i])
        if i == 0:
            new_Y = y
        else:
            new_Y = np.hstack([new_Y,y])
        dic_list.append(dic)
 
    return new_Y,dic_list
 
def split_test_as_valid(test_y_c):
    
    unique_value = np.unique(test_y_c)
    idx = np.ones(test_y_c.shape[0])
    for i_uni,e_uni in enumerate(unique_value):
        a = idx[np.where(test_y_c[:,0]==e_uni)]
        idx[np.where(test_y_c[:,0]==e_uni)] = a*np.round(np.random.rand(len(a)))
        
    idx_pos = idx
    idx_neg = 1-idx
    
    return idx_pos.astype(bool),idx_neg.astype(bool)

 

def split_train_test(train_y_a,test_rate = 0.2):
    
    user_id_set = np.unique(train_y_a[:,0])
    user_age_set = np.array([train_y_a[train_y_a[:,0]==e,1][0] for e in user_id_set])
    test_user_id_set = []
    for age in np.unique(user_age_set):
      
        user_id_subset = user_id_set[user_age_set == age].copy()
        np.random.shuffle(user_id_subset)
        test_user_id_set.extend(user_id_subset[:int(np.ceil(len(user_id_subset)*test_rate))])
    print 'user_id_set,test_user_id_set',user_id_set,test_user_id_set
    train_user_id_set = np.setdiff1d(user_id_set,test_user_id_set)
 
    return train_user_id_set,test_user_id_set
 
def argmin_mean_FRR_st_FAR(label_test, prob,exp_far=0.01,pen_far=10.):
    
    def f(e):
        label_out = prob > e
        far = FAR_score(label_test, label_out)
        frr = FRR_score(label_test, label_out)
        return frr + pen_far * np.maximum(0,far-exp_far)
    res = minimize_scalar(f, bounds=(prob.min(), prob.max()), method='bounded')
     
    return res.x

def argmin_fixFRR(label_test, prob,exp_frr=0.05):
    
    def f(e):
        label_out = prob > e
        far = FAR_score(label_test, label_out)
        frr = FRR_score(label_test, label_out)
        return np.abs(frr - exp_frr)
    res = minimize_scalar(f, bounds=(prob.min(), prob.max()), method='bounded')
     
    return res.x

def argmin_mean_FAR_FRR(label_test, prob):
    
    def f(e):
        label_out = prob > e
        far = FAR_score(label_test, label_out)
        frr = FRR_score(label_test, label_out)
        return np.abs(far-frr)
    res = minimize_scalar(f, bounds=(prob.min(), prob.max()), method='bounded')
     
    return res.x
   
def auc_MTL(label_test, prob):
    if np.ndim(prob) == 1:
        auc_all = roc_auc_score(label_test, prob)

        return auc_all, auc_all
    else:
        valid_task_id = []
        for i in range(label_test.shape[1]):
            if len(np.unique(label_test[: ,i])) > 1:
                valid_task_id.append(i)
        task_num = len(valid_task_id)
        auc_all = []
        for i_order in valid_task_id:
            auc_all.append(roc_auc_score(label_test[:, i_order], prob[:, i_order]))
        auc_all = np.array(auc_all)

        return np.mean(auc_all), auc_all
     
def FAR_score(y_true,y_pred):
    
    return np.sum(y_pred[np.where(y_true==0)])/float(np.sum(y_true==0))
 
def FRR_score(y_true,y_pred):
    
    return np.sum(y_pred[np.where(y_true==1)]==0)/float(np.sum(y_true==1))
 
def evaluate_result_valid_simple(label_test, prob,i,str_input):
    label_out = np.round(prob)
    mean_auc, auc_all = auc_MTL(label_test, prob)
 
    acc = np.mean(np.argmax(label_test,axis=1) == np.argmax(prob,axis=1))
    
    Y_test = label_test
    train_labels = np.unique(np.argmax(Y_test,axis=1))
    n_user_train = len(train_labels)


    print('%s: step %d, auc %g, acc %g, n_eval %d' % (str_input, i,  np.mean(auc_all),acc,n_user_train))
#     print('%s: step %d, [%g, %g, %g, %g, %g]' % (str_input, i,  np.mean(auc_all), acc,np.mean(ACC_ALL),np.mean(FAR_ALL),np.mean(FRR_ALL)))
    print auc_all 
    
    
    return auc_all
     
def evaluate_result_valid(label_test, prob,i,str_input):
    
    mean_auc, auc_all = auc_MTL(label_test, prob)
 
    Y_test = label_test
    train_labels = np.unique(np.argmax(Y_test,axis=1))
    n_user_train = len(train_labels)

    dic_th = {}
    FAR_ALL = np.zeros(n_user_train)
    FRR_ALL = np.zeros(n_user_train)
    ACC_ALL = np.zeros(n_user_train)
    for i_user,idx in enumerate(train_labels):
        dic_th[idx] = argmin_mean_FAR_FRR(Y_test[:,idx], prob[:,idx])
        label_out  = prob[:,idx] > dic_th[idx]
        FAR_ALL[i_user] = FAR_score(Y_test[:,idx],label_out )
        FRR_ALL[i_user] = FRR_score(Y_test[:,idx],label_out )
        ACC_ALL[i_user] = np.mean(Y_test[:,idx] == label_out )
        
    acc = np.mean(np.argmax(Y_test,axis=1) == np.argmax(prob,axis=1))


    print('%s: step %d, auc %g, FAR %g, FRR %g, acc %g, acc_avg %g, n_eval %d' % (str_input, i,  np.mean(auc_all), np.mean(FAR_ALL),np.mean(FRR_ALL),acc,np.mean(ACC_ALL),n_user_train))
 
    return dic_th,np.mean(auc_all)

def evaluate_result_test(label_test, prob,i,str_input,dic_th):
 
    mean_auc, auc_all = auc_MTL(label_test, prob)

    Y_test = label_test
    train_labels = np.unique(np.argmax(Y_test,axis=1))
    n_user_train = len(train_labels)

    FAR_ALL = np.zeros(n_user_train)
    FRR_ALL = np.zeros(n_user_train)
    ACC_ALL = np.zeros(n_user_train)
    for i_user,idx in enumerate(train_labels):
        label_out  = prob[:,idx] > dic_th.get(idx,0.5)
        FAR_ALL[i_user] = FAR_score(Y_test[:,idx],label_out )
        FRR_ALL[i_user] = FRR_score(Y_test[:,idx],label_out )
        ACC_ALL[i_user] = np.mean(Y_test[:,idx] == label_out )
        
    acc = np.mean(np.argmax(Y_test,axis=1) == np.argmax(prob,axis=1))


    print('%s: step %d, auc %g, FAR %g, FRR %g, acc %g, acc_avg %g, n_eval %d' % (str_input, i,  np.mean(auc_all), np.mean(FAR_ALL),np.mean(FRR_ALL),acc,np.mean(ACC_ALL),n_user_train))
#     print('%s: step %d, [%g, %g, %g, %g, %g]' % (str_input, i,  np.mean(auc_all), acc,np.mean(ACC_ALL),np.mean(FAR_ALL),np.mean(FRR_ALL)))
#     print auc_all

    return
 
def evaluate_result_valid_fixFAR(label_test, prob,i,str_input):
  
    mean_auc, auc_all = auc_MTL(label_test, prob)
     
    Y_test = label_test
    train_labels = np.unique(np.argmax(Y_test,axis=1))
    n_user_train = len(train_labels)

    dic_th = {}
    FAR_ALL = np.zeros(n_user_train)
    FRR_ALL = np.zeros(n_user_train)
    ACC_ALL = np.zeros(n_user_train)
    for i_user,idx in enumerate(train_labels):

        dic_th[idx] = argmin_mean_FRR_st_FAR(Y_test[:,idx], prob[:,idx])
        
        label_out  = prob[:,idx] > dic_th[idx]
        FAR_ALL[i_user] = FAR_score(Y_test[:,idx],label_out )
        FRR_ALL[i_user] = FRR_score(Y_test[:,idx],label_out )
        ACC_ALL[i_user] = np.mean(Y_test[:,idx] == label_out )
        
    acc = np.mean(np.argmax(Y_test,axis=1) == np.argmax(prob,axis=1))


    print('%s: step %d, auc %g, FAR %g, FRR %g, acc %g, acc_avg %g, n_eval %d' % (str_input, i,  np.mean(auc_all), np.mean(FAR_ALL),np.mean(FRR_ALL),acc,np.mean(ACC_ALL),n_user_train))
#     print('%s: step %d, [%g, %g, %g, %g, %g]' % (str_input, i,  np.mean(auc_all), acc,np.mean(ACC_ALL),np.mean(FAR_ALL),np.mean(FRR_ALL)))
#     print auc_all 
    
    
    return dic_th,np.mean(auc_all)

import multiprocessing
from multiprocessing import Pool
from multiprocessing import Process

import multiprocessing
from itertools import product
from contextlib import contextmanager

def sub_fun(i_user,Y_test,prob):
    idx = i_user
    dic_th = argmin_mean_FRR_st_FAR(Y_test, prob)

    label_out = prob > dic_th
    FAR_ALL = FAR_score(Y_test,label_out)
    FRR_ALL = FRR_score(Y_test,label_out)
    ACC_ALL = np.mean(Y_test == label_out)
    return FAR_ALL,FRR_ALL,ACC_ALL,dic_th

def sub_fun_unpack(args):
    return sub_fun(*args)

@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()


def evaluate_result_valid_fixFAR_pool(test_Y, prob,i,str_input,num_core=20):
      
 
    train_labels = np.unique(test_Y)
    n_user_train = len(train_labels)

#     FAR_ALL = np.zeros(n_user_train)
#     FRR_ALL = np.zeros(n_user_train)
#     ACC_ALL = np.zeros(n_user_train)
 
    with poolcontext(processes=num_core) as pool:
        results = pool.map(sub_fun_unpack, [(i_user,(test_Y==i_user).astype(int),prob[:,i_user]) for i_user in train_labels])
    
    results = np.array(results)
    
    FAR_ALL = results[:,0]
    FRR_ALL = results[:,1]
    ACC_ALL = results[:,2]
    dic_th =  results[:,3]
    
        
    acc = np.mean(test_Y == np.argmax(prob,axis=1))


    print('%s: step %d, FAR %g, FRR %g, acc %g, acc_avg %g, n_eval %d' % (str_input, i, np.mean(FAR_ALL),np.mean(FRR_ALL),acc,np.mean(ACC_ALL),n_user_train))
 
    return dic_th,1-np.mean(FRR_ALL)

def evaluate_result_valid_fixFAR_nopool(test_Y, prob,i,str_input,num_core=20):
      
 
    train_labels = np.unique(test_Y)
    n_user_train = len(train_labels)

    FAR_ALL = np.zeros(n_user_train)
    FRR_ALL = np.zeros(n_user_train)
    ACC_ALL = np.zeros(n_user_train)
    dic_th  = np.zeros(n_user_train)
    start_time = time_simple.clock() 
    for i_user in train_labels:
        if i_user % 1000 == 0:
            print i_user
            end_time = time_simple.clock()
            print 'testing time: %.3f' % (end_time - start_time)
            start_time = time_simple.clock() 
            
        FAR_ALL[i_user],FRR_ALL[i_user],ACC_ALL[i_user],dic_th[i_user] = sub_fun(i_user,(test_Y==i_user).astype(int),prob[:,i_user])
      
    acc = np.mean(test_Y == np.argmax(prob,axis=1))


    print('%s: step %d, FAR %g, FRR %g, acc %g, acc_avg %g, n_eval %d' % (str_input, i, np.mean(FAR_ALL),np.mean(FRR_ALL),acc,np.mean(ACC_ALL),n_user_train))
 
    return dic_th,1-np.mean(FRR_ALL)

 

def sub_fun_test(i_user,Y_test,prob,dic_th):
    idx = i_user
 
    label_out = prob > dic_th 
    FAR_ALL = FAR_score(Y_test,label_out)
    return FAR_ALL 

def sub_fun_test_unpack(args):
    return sub_fun_test(*args)

def evaluate_result_test_attack_pool(prob,i,str_input,dic_th,num_core=20):
 
    
 
    n,n_user_train = prob.shape[0],prob.shape[1]

     
        
    with poolcontext(processes=num_core) as pool:
        results = pool.map(sub_fun_test_unpack, [(i_user,np.zeros(n),prob[:,i_user],dic_th[i_user]) for i_user in range(n_user_train)])
      
    results = np.array(results)
    FAR_ALL = results 
    


    print('%s: step %d, FAR %g, n_eval %d' % (str_input, i,   np.mean(FAR_ALL),n_user_train))
     
 

    return

def evaluate_result_test_attack_nopool(prob,i,str_input,dic_th,num_core=20):
  
    n,n_user_train = prob.shape[0],prob.shape[1]
    
    FAR_ALL = np.zeros(n_user_train)
    start_time = time_simple.clock() 
    for i_user in range(n_user_train):
        if i_user % 1000 == 0:
            print i_user
            end_time = time_simple.clock()
            print 'testing time: %.3f' % (end_time - start_time)
            start_time = time_simple.clock() 

        FAR_ALL[i_user] = sub_fun_test(i_user,np.zeros(n),prob[:,i_user],dic_th[i_user])
          
    print('%s: step %d, FAR %g, n_eval %d' % (str_input, i,   np.mean(FAR_ALL),n_user_train))
     
    return

def evaluate_result_test_attack_real_attack(label_test, prob,i,str_input,dic_th):
     
   

    Y_test = label_test
 
    
    test_labels = np.unique(np.argmax(label_test,axis=1))
    n_user_train = len(test_labels)
    FRR_ALL = np.zeros(n_user_train)
   
    for i_user,idx in enumerate(test_labels):
      
        label_out= prob[:,idx] > dic_th[idx]
        FRR_ALL[i_user] = FRR_score(Y_test[:,idx],label_out)
      
         


    print('%s: step %d, FAR %g, n_eval %d' % (str_input, i,   1-np.mean(FRR_ALL),n_user_train))
     
 

    return

def evaluate_result_test_attack(label_test, prob,i,str_input,dic_th):
     
   

    Y_test = label_test
 
    n_user_train = label_test.shape[1]

    FAR_ALL = np.zeros(n_user_train)
   
    for i_user in range(n_user_train):
        idx =  i_user
        label_out= prob[:,idx] > dic_th[idx]
        FAR_ALL[i_user] = FAR_score(Y_test[:,idx],label_out)
      
         


    print('%s: step %d, FAR %g, n_eval %d' % (str_input, i,   np.mean(FAR_ALL),n_user_train))
     
 

    return



def evaluate_result_age(label_test, prob,i,str_input,age_th=16):
     
    label_test_binary = (label_test>age_th).astype(int)
    auc = roc_auc_score(label_test_binary, prob)
    th = argmin_mean_FAR_FRR(label_test_binary, prob)
        
    label_out  = prob > th
    far = FAR_score(label_test_binary,label_out)
    frr = FRR_score(label_test_binary,label_out)
    acc = np.mean(label_test_binary==label_out)
     
    mae = np.mean(np.abs(label_test-prob))


    print('%s: step %d, auc %g, FAR %g, FRR %g, acc %g, mae %g' % (str_input, i,  auc, far,frr,acc,mae))
#     print('%s: step %d, [%g, %g, %g, %g, %g]' % (str_input, i,  np.mean(auc_all), acc,np.mean(ACC_ALL),np.mean(FAR_ALL),np.mean(FRR_ALL)))
  
    
    
    return th,auc, far,frr,acc,mae

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def my_zscore(X):
    mean_X = np.mean(X, 0)
    std_X = np.std(X, 0)
    std_X[np.where(std_X == 0.0)] = 1.0
    return (X - mean_X) / std_X, mean_X, std_X


def my_zscore_test(X, mean_X, std_X):
    std_X[np.where(std_X == 0.0)] = 1.0
    return (X - mean_X) / std_X




def mat_dist(qf,gf):
    m,n = qf.shape[0],gf.shape[0]
    distmat = (qf**2).sum(axis=1)[:,np.newaxis] * np.ones((1,n)) + \
               np.ones((m,1)) * (gf**2).sum(axis=1)[np.newaxis,:] 
    return np.sqrt(np.maximum(0.0,distmat - 2 * np.dot(qf,gf.T))) 


def sub_fun_dist(i_user,test_X,Y_test,train_X,argmin_fun=argmin_mean_FRR_st_FAR):
    
    dist = mat_dist(test_X,train_X)
    prob = - np.min(dist,axis=1)
    
    
    idx = i_user
    dic_th = argmin_fun(Y_test, prob)

    label_out = prob > dic_th
    FAR_ALL = FAR_score(Y_test,label_out)
    FRR_ALL = FRR_score(Y_test,label_out)
    ACC_ALL = np.mean(Y_test == label_out)

    return FAR_ALL,FRR_ALL,ACC_ALL,dic_th
 



def sub_fun_dist_unpack(args):
    return sub_fun_dist(*args)

@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()
    
def eval_distBased_valid_fixFAR_pool(train_X,train_Y,test_X,test_Y,i,str_input,num_core=20):
     
        
    dic_th = {}
 
    test_labels = np.unique(test_Y)
    train_labels = np.unique(train_Y)
    final_labels = np.intersect1d(train_labels,test_labels)
    
    n_user_train = len(final_labels)
     
#     print 'np.any(np.isnan(train_X)),np.any(np.isnan(test_X))',np.any(np.isnan(train_X)),np.any(np.isnan(test_X))
 
    with poolcontext(processes=num_core) as pool:
        results = pool.map(sub_fun_dist_unpack, [(i_user,test_X,(test_Y==i_user).astype(int),train_X[train_Y==i_user,:]) for i_user in final_labels])
    
    results = np.array(results)
    
    FAR_ALL = results[:,0]
    FRR_ALL = results[:,1]
    ACC_ALL = results[:,2]
    dic_th =  results[:,3]
    
         


    print('%s: step %d, FAR %g, FRR %g, acc_avg %g, n_eval %d' % (str_input, i, np.mean(FAR_ALL),np.mean(FRR_ALL),np.mean(ACC_ALL),n_user_train))
#     print('%s: step %d, [%g, %g, %g, %g, %g]' % (str_input, i,  np.mean(auc_all), acc,np.mean(ACC_ALL),np.mean(FAR_ALL),np.mean(FRR_ALL)))
   
  
    
    return dic_th,1-np.mean(FRR_ALL)


def sub_fun_dist_testattack(i_user,test_X,Y_test,test_X_attack,Y_test_attack,train_X,argmin_fun=argmin_mean_FRR_st_FAR,exp_frr=0.1):
    
    dist = mat_dist(test_X,train_X)
    prob = - np.min(dist,axis=1)
    dist_attack = mat_dist(test_X_attack,train_X)
    prob_attack = - np.min(dist_attack,axis=1)
    
    
    idx = i_user
    dic_th = argmin_fun(Y_test, prob,exp_frr=exp_frr)

    label_out = prob > dic_th
    label_out_attack = prob_attack > dic_th
    
    FAR_ALL = FAR_score(Y_test,label_out)
    FRR_ALL = FRR_score(Y_test,label_out)
    FAR_ALL_attack = 1 - FRR_score(Y_test_attack,label_out_attack)
    sum_1_label_test = np.sum(Y_test)
    sum_1_label_out = np.sum(label_out*Y_test)
    sum_1_label_out_attack = np.sum(label_out_attack*Y_test_attack)
    sum_1_label_test_attack = np.sum(Y_test_attack)
    return FAR_ALL,FRR_ALL,FAR_ALL_attack,dic_th,sum_1_label_test,sum_1_label_out,sum_1_label_out_attack,sum_1_label_test_attack

def eval_distBased_testattack(train_X,train_Y,test_X,test_Y,test_X_attack,test_Y_attack,i,str_input,num_core=20,argmin_fun=argmin_mean_FRR_st_FAR,exp_frr=0.1):
  

    n_user_train = train_Y.shape[1]


    FAR_ALL = np.zeros(n_user_train)
    FRR_ALL = np.zeros(n_user_train)
    FAR_ALL_attack = np.zeros(n_user_train)
    dic_th  = np.zeros(n_user_train)
    sum_1_label_test = np.zeros(n_user_train)
    sum_1_label_out = np.zeros(n_user_train)
    sum_1_label_out_attack = np.zeros(n_user_train)
    sum_1_label_test_attack = np.zeros(n_user_train)
    
    start_time = time_simple.clock() 
    for i_user in range(n_user_train):
        FAR_ALL[i_user],FRR_ALL[i_user],FAR_ALL_attack[i_user],dic_th[i_user],sum_1_label_test[i_user],sum_1_label_out[i_user],sum_1_label_out_attack[i_user],sum_1_label_test_attack[i_user] = sub_fun_dist_testattack(i_user,test_X,test_Y[:,i_user],test_X_attack,test_Y_attack[:,i_user],train_X[train_Y[:,i_user]==1,:],argmin_fun=argmin_fun,exp_frr=exp_frr)
 
    print('%s: step %d, FAR %g, FRR %g, FAR_attack %g, n_eval_far %d, n_eval_frr %d, n_eval_far_attack %d' % (str_input, i, np.mean(FAR_ALL[np.invert(np.isnan(FAR_ALL))]),np.mean(FRR_ALL[np.invert(np.isnan(FRR_ALL))]),np.mean(FAR_ALL_attack[np.invert(np.isnan(FAR_ALL_attack))]),np.sum(np.invert(np.isnan(FAR_ALL))),np.sum(np.invert(np.isnan(FRR_ALL))),np.sum(np.invert(np.isnan(FAR_ALL_attack)))))

    print('macro:\trecall %g,\t\t\tattack %g' %(1. - np.mean(FRR_ALL[np.invert(np.isnan(FRR_ALL))]),np.mean(FAR_ALL_attack[np.invert(np.isnan(FAR_ALL_attack))])))
    print('micro:\trecall %g(%d/%d),\tattack %g(%d/%d)' %(np.sum(sum_1_label_out)/np.sum(sum_1_label_test),np.sum(sum_1_label_out),np.sum(sum_1_label_test),
                                                          np.sum(sum_1_label_out_attack)/np.sum(sum_1_label_test_attack),
                                                         np.sum(sum_1_label_out_attack),np.sum(sum_1_label_test_attack)))
     
        


def eval_distBased_valid_fixFAR_nopool(train_X,train_Y,test_X,test_Y,i,str_input,num_core=20,argmin_fun=argmin_mean_FRR_st_FAR):
      
 
    if np.ndim(train_Y)==2 and train_Y.shape[1] > 1:
        

        n_user_train = train_Y.shape[1]


        FAR_ALL = np.zeros(n_user_train)
        FRR_ALL = np.zeros(n_user_train)
        ACC_ALL = np.zeros(n_user_train)
        dic_th  = np.zeros(n_user_train)
        sum_1_label_test = np.zeros(n_user_train)
        sum_1_label_out = np.zeros(n_user_train)
        start_time = time_simple.clock() 
        for i_user in range(n_user_train):
    #         if i_user % 1000 == 0:
    #             print i_user
    #             end_time = time_simple.clock()
    #             print 'testing time: %.3f' % (end_time - start_time)
    #             start_time = time_simple.clock() 

            FAR_ALL[i_user],FRR_ALL[i_user],ACC_ALL[i_user],dic_th[i_user],sum_1_label_test[i_user],sum_1_label_out[i_user] = sub_fun_dist(i_user,test_X,test_Y[:,i_user],train_X[train_Y[:,i_user]==1,:],argmin_fun=argmin_fun)




        print('%s: step %d, FAR %g, FRR %g, acc_avg %g, n_eval %d' % (str_input, i, np.mean(FAR_ALL),np.mean(FRR_ALL),np.mean(ACC_ALL),n_user_train))
    #     print('%s: step %d, [%g, %g, %g, %g, %g]' % (str_input, i,  np.mean(auc_all), acc,np.mean(ACC_ALL),np.mean(FAR_ALL),np.mean(FRR_ALL)))

        print('macro:\trecall %g' %(1. - np.mean(FRR_ALL[np.invert(np.isnan(FRR_ALL))])))
        print('micro:\trecall %g(%d/%d)' %(np.sum(sum_1_label_out)/np.sum(sum_1_label_test),np.sum(sum_1_label_out),np.sum(sum_1_label_test)))
        
    else:
        test_labels = np.unique(test_Y)
        train_labels = np.unique(train_Y)
        final_labels = np.intersect1d(train_labels,test_labels)

        n_user_train = len(final_labels)


        FAR_ALL = np.zeros(n_user_train)
        FRR_ALL = np.zeros(n_user_train)
        ACC_ALL = np.zeros(n_user_train)
        dic_th  = np.zeros(n_user_train)
        sum_1_label_test = np.zeros(n_user_train)
        sum_1_label_out = np.zeros(n_user_train)
        start_time = time_simple.clock() 
        for i_user in final_labels:
    #         if i_user % 1000 == 0:
    #             print i_user
    #             end_time = time_simple.clock()
    #             print 'testing time: %.3f' % (end_time - start_time)
    #             start_time = time_simple.clock() 

            FAR_ALL[i_user],FRR_ALL[i_user],ACC_ALL[i_user],dic_th[i_user],sum_1_label_test[i_user],sum_1_label_out[i_user] = sub_fun_dist(i_user,test_X,(test_Y==i_user).astype(int),train_X[train_Y==i_user,:],argmin_fun=argmin_fun)




        print('%s: step %d, FAR %g, FRR %g, acc_avg %g, n_eval %d' % (str_input, i, np.mean(FAR_ALL),np.mean(FRR_ALL),np.mean(ACC_ALL),n_user_train))
    #     print('%s: step %d, [%g, %g, %g, %g, %g]' % (str_input, i,  np.mean(auc_all), acc,np.mean(ACC_ALL),np.mean(FAR_ALL),np.mean(FRR_ALL)))

        print('macro:\trecall %g' %(1. - np.mean(FRR_ALL[np.invert(np.isnan(FRR_ALL))])))
        print('micro:\trecall %g(%d/%d)' %(np.sum(sum_1_label_out)/np.sum(sum_1_label_test)))
    
  
    
    return dic_th,1-np.mean(FRR_ALL)

def eval_train_test_correctness(train_X,train_Y,test_X,test_Y):
    
    test_labels = np.unique(test_Y)
    train_labels = np.unique(train_Y)
    final_labels = np.intersect1d(train_labels,test_labels)
    
    n_user_train = len(final_labels)
    
    n = test_Y.shape[0]
     
 
    dist_list = np.zeros(n_user_train)
    start_time = time_simple.time()
    for i_user in final_labels:
        if i_user % 1000 == 0:
            print i_user
            end_time = time_simple.time()
            print 'testing time: %.3f' % (end_time - start_time)
            start_time = time_simple.time()
  
        test_X_pos = test_X[test_Y==i_user,:]
        
        train_X_i = train_X[train_Y==i_user,:]
        
        dist = mat_dist(test_X_pos,train_X_i)
        dist_list[i_user] = np.min(dist)
         
 
    return dist_list

def eval_distBased_valid_fixFAR_nopool_negDownsample(train_X,train_Y,test_X,test_Y,i,str_input,num_core=20,n_neg = 10000,flag_print=True):
    
    test_labels = np.unique(test_Y)
    train_labels = np.unique(train_Y)
    final_labels = np.intersect1d(train_labels,test_labels)
    
    n_user_train = len(final_labels)
    
    n = test_Y.shape[0]
    
    idx_sample = np.random.rand(n) < float(n_neg)/float(n)
    test_X_for_neg = test_X[idx_sample,:]
    test_Y_for_neg = test_Y[idx_sample]
 
    FAR_ALL = np.zeros(n_user_train)
    FRR_ALL = np.zeros(n_user_train)
    ACC_ALL = np.zeros(n_user_train)
    dic_th  = np.zeros(n_user_train)
    start_time = time_simple.clock() 
    for i_user,e_user in enumerate(final_labels):
#         if i_user % 1000 == 0:
#             print i_user
#             end_time = time_simple.time()
#             print 'testing time: %.3f' % (end_time - start_time)
#             start_time = time_simple.time()
            
        idx_i = test_Y==e_user
        idx_i_neg = test_Y_for_neg != e_user
        test_X_pos = test_X[idx_i,:]
        test_X_neg = test_X_for_neg[idx_i_neg,:]
        test_X_proc = np.vstack([test_X_pos,test_X_neg])
        test_Y_proc = np.hstack([np.ones(test_X_pos.shape[0]),np.zeros(test_X_neg.shape[0])])
            
        FAR_ALL[i_user],FRR_ALL[i_user],ACC_ALL[i_user],dic_th[i_user] = sub_fun_dist(i_user,test_X_proc,test_Y_proc,train_X[train_Y==e_user,:])

    if flag_print:
        print('%s: step %d, FAR %g, FRR %g, acc_avg %g, n_eval %d' % (str_input, i, np.mean(FAR_ALL),np.mean(FRR_ALL),np.mean(ACC_ALL),n_user_train))
 
    return dic_th,1-np.mean(FRR_ALL),FAR_ALL,FRR_ALL

def eval_angleBased_valid_fixFAR_nopool_negDownsample_sameDeviceType(train_X,train_Y,test_X,test_Y,i,str_input,num_core=20,n_neg = 10000,flag_print=True):
    
    test_labels = np.unique(test_Y[:,0])
    train_labels = np.unique(train_Y[:,0])
    final_labels = np.intersect1d(train_labels,test_labels)
    
    n_user_train = len(final_labels)
    
    n = test_Y.shape[0]
 
    FAR_ALL = np.zeros(n_user_train)
    FRR_ALL = np.zeros(n_user_train)
    ACC_ALL = np.zeros(n_user_train)
    dic_th  = np.zeros(n_user_train)
    start_time = time_simple.clock() 
    for i_user,e_user in enumerate(final_labels):
        
        idx_i = test_Y[:,0]==e_user
        this_DeviceType = test_Y[idx_i,1]
        test_X_pos = test_X[idx_i,:]
        
 
        idx_sample_sameDeviceType = test_Y[:,1]==this_DeviceType[0]
        test_X_for_neg = test_X[idx_sample_sameDeviceType,:]
        test_Y_for_neg = test_Y[idx_sample_sameDeviceType,:]
        n = test_Y_for_neg.shape[0]
        
        idx_sample = np.random.rand(n) < float(n_neg)/float(n)
        test_X_for_neg = test_X_for_neg[idx_sample,:]
        test_Y_for_neg = test_Y_for_neg[idx_sample,0]
            
        
        idx_i_neg = test_Y_for_neg != e_user  
        test_X_neg = test_X_for_neg[idx_i_neg,:]
        
        test_X_proc = np.vstack([test_X_pos,test_X_neg])
        test_Y_proc = np.hstack([np.ones(test_X_pos.shape[0]),np.zeros(test_X_neg.shape[0])])
            
        FAR_ALL[i_user],FRR_ALL[i_user],ACC_ALL[i_user],dic_th[i_user] = sub_fun_angle(i_user,test_X_proc,test_Y_proc,train_X[train_Y[:,0]==e_user,:])

    if flag_print:
        print('%s: step %d, FAR %g, FRR %g, acc_avg %g, n_eval %d' % (str_input, i, np.mean(FAR_ALL),np.mean(FRR_ALL),np.mean(ACC_ALL),n_user_train))
    
    FAR_ALL = FAR_ALL[np.invert(np.isnan(FAR_ALL))]
    FRR_ALL = FRR_ALL[np.invert(np.isnan(FRR_ALL))]
    
    return dic_th,1-np.mean(FRR_ALL),FAR_ALL,FRR_ALL

def eval_distBased_valid_fixFAR_nopool_negDownsample_sameDeviceType(train_X,train_Y,test_X,test_Y,i,str_input,num_core=20,n_neg = 10000,flag_print=True):
    
    test_labels = np.unique(test_Y[:,0])
    train_labels = np.unique(train_Y[:,0])
    final_labels = np.intersect1d(train_labels,test_labels)
    
    n_user_train = len(final_labels)
    
    n = test_Y.shape[0]
 
    FAR_ALL = np.zeros(n_user_train)
    FRR_ALL = np.zeros(n_user_train)
    ACC_ALL = np.zeros(n_user_train)
    dic_th  = np.zeros(n_user_train)
    start_time = time_simple.clock() 
    for i_user,e_user in enumerate(final_labels):
        
        idx_i = test_Y[:,0]==e_user
        this_DeviceType = test_Y[idx_i,1]
        test_X_pos = test_X[idx_i,:]
        
 
        idx_sample_sameDeviceType = test_Y[:,1]==this_DeviceType[0]
        test_X_for_neg = test_X[idx_sample_sameDeviceType,:]
        test_Y_for_neg = test_Y[idx_sample_sameDeviceType,:]
        n = test_Y_for_neg.shape[0]
        
        idx_sample = np.random.rand(n) < float(n_neg)/float(n)
        test_X_for_neg = test_X_for_neg[idx_sample,:]
        test_Y_for_neg = test_Y_for_neg[idx_sample,0]
            
        
        idx_i_neg = test_Y_for_neg != e_user  
        test_X_neg = test_X_for_neg[idx_i_neg,:]
        
        test_X_proc = np.vstack([test_X_pos,test_X_neg])
        test_Y_proc = np.hstack([np.ones(test_X_pos.shape[0]),np.zeros(test_X_neg.shape[0])])
            
        FAR_ALL[i_user],FRR_ALL[i_user],ACC_ALL[i_user],dic_th[i_user] = sub_fun_dist(i_user,test_X_proc,test_Y_proc,train_X[train_Y[:,0]==e_user,:])

    if flag_print:
        print('%s: step %d, FAR %g, FRR %g, acc_avg %g, n_eval %d' % (str_input, i, np.mean(FAR_ALL),np.mean(FRR_ALL),np.mean(ACC_ALL),n_user_train))
    
    FAR_ALL = FAR_ALL[np.invert(np.isnan(FAR_ALL))]
    FRR_ALL = FRR_ALL[np.invert(np.isnan(FRR_ALL))]
    
    return dic_th,1-np.mean(FRR_ALL),FAR_ALL,FRR_ALL


def sub_fun_read_data_metric(file_path):
    
    df = pd.read_csv(file_path,header=0)   
    mat = df.values
        
    return mat

def sub_fun_read_data_metric_unpack(args):
    return sub_fun_read_data_metric(*args)

@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()

def read_data_metric_pool(file_list,flag_zscore=False,num_core=20,flag_delete_anomaly=False,flag_multilabel=False,mean_x = None,std_x=None):
    
    len_file_list = len(file_list)
    n_core = int(np.minimum(len_file_list,num_core))
    if n_core>1:
        with poolcontext(processes=n_core) as pool:
            results = pool.map(sub_fun_read_data_metric, [file_path for file_path in file_list])
        mat = np.vstack(results)
    else:
        mat = sub_fun_read_data_metric(file_list[0])
    
    
    
#     results = np.array(results)
    
    if flag_multilabel:
        unknown_type = mat[:,1]
        mat = mat[unknown_type!='-1',:]
        Y = mat[:,0:2]
    else:
        Y = mat[:,0][:,np.newaxis] 
    X = mat[:,-191:] 
    
#     print(len(X))  
    if flag_delete_anomaly:
        anomaly = mat[:,2]
        X = X[anomaly==0,:]
        Y = Y[anomaly==0,:]
    
#     print(len(X)) 

    nan_idx = np.sum(np.isnan(X),axis=1) == 0
    X = X[nan_idx,:]
    Y = Y[nan_idx,:]
    
#     print(len(X)) 
  
    if flag_zscore:
#         if not mean_x:
#             save_root_path = 'extracted_transformer_pickle'
#             write_path = 'param_wechat_ds20190120_fold_4_step_100000.p'
#             param_dic = load_param(save_root_path,write_path)
#             mean_x = param_dic['mean_x']
#             std_x = param_dic['std_x']
        
        X = my_zscore_test(X,mean_x,std_x)
 
    return  X,Y

def read_h5(file_path):
    f = h5py.File(file_path,'r')
    x = np.array(f['X'])
    f.close()
    return x
def read_csv(file_path):
    df = pd.read_csv(file_path,header=None)   
    x = df.values
    return x

def read_h5andcsv(file_path_csv,file_path_h5,flag_delete_anomaly=False,flag_multilabel=False,flag_typecoded = True,flag_includeTime=False,flag_191feat = False,flag_select_device_type=False,selected_device_type=None,flag_no_pressure=False):
    
    df = pd.read_csv(file_path_csv,header=None)   
    mat_str = df.values
    
    f = h5py.File(file_path_h5,'r')
    
    
    model_type = mat_str[:,1]
    if flag_typecoded:
        dype_values = ID_TO_PHONE_MODEL_MAPPING.values()
        model_use = np.array(map(lambda x: x in dype_values,model_type))
#             np.array([e_type in dype_values for e_type in model_type])
    else:
        model_use = np.array(map(lambda x: x in ID_TO_PHONE_MODEL_MAPPING,model_type))
#             np.array([e_type in ID_TO_PHONE_MODEL_MAPPING for e_type in model_type])
        mat_str[:,1] = np.array(map(lambda x: ID_TO_PHONE_MODEL_MAPPING.get(x,'-1'),model_type))
#             np.array([ID_TO_PHONE_MODEL_MAPPING[e_type] for e_type in model_type])
 
        
    if flag_select_device_type:
        device_type_use = mat_str[:,1] == selected_device_type
        if flag_191feat:
            mat_float = np.array(f['X'][device_type_use*model_use,:-(974-191)])
        else:
            mat_float = np.array(f['X'][device_type_use*model_use,:])
        mat_str = mat_str[device_type_use*model_use,:]
    else:
        if flag_191feat:
            mat_float = np.array(f['X'][model_use,:-(974-191)])
        else:
            mat_float = np.array(f['X'][model_use,:])
        mat_str = mat_str[model_use,:]
 
    if not flag_multilabel:
        mat_str = mat_str[:,0][:,np.newaxis] 
        
        
    f.close()
    
    Y = mat_str
    
     
    
    if flag_includeTime:   
        if flag_no_pressure:
            X = np.hstack([mat_float[:,:2],mat_float[:,2+12:]])
        else:
            X = mat_float
    else:
        if flag_no_pressure:
            X = mat_float[:,2+12:]
        else:
            X = mat_float[:,2:]
 
    if flag_delete_anomaly:
        anomaly = mat_float[:,0]
        X = X[anomaly==0,:]
        Y = Y[anomaly==0,:]
        
#     print 'anomaly rate = %g(%d/%d)' % (np.sum(anomaly>0)/X.shape[0],np.sum(anomaly>0),X.shape[0])
 
    nan_idx = np.sum(np.isnan(X),axis=1) == 0
    X = X[nan_idx,:]
    Y = Y[nan_idx,:]
 
    return X,Y
    

def read_h5andcsv_unpack(args):
    return read_h5andcsv(*args)

ID_TO_PHONE_MODEL_MAPPING = {
        'iPhone1,1': 'iPhone',

        'iPhone1,2': 'iPhone 3G',

        'iPhone2,1': 'iPhone 3GS',

        'iPhone3,1': 'iPhone 4',
        'iPhone3,2': 'iPhone 4',
        'iPhone3,3': 'iPhone 4',

        'iPhone4,1': 'iPhone 4S',

        'iPhone5,1': 'iPhone 5',
        'iPhone5,2': 'iPhone 5',

        'iPhone5,3': 'iPhone 5c',
        'iPhone5,4': 'iPhone 5c',

        'iPhone6,1': 'iPhone 5s',
        'iPhone6,2': 'iPhone 5s',

        'iPhone7,2': 'iPhone 6',
        'iPhone7,1': 'iPhone 6 Plus',

        'iPhone8,1': 'iPhone 6s',
        'iPhone8,2': 'iPhone 6s Plus',
        'iPhone8,4': 'iPhone SE',

        'iPhone9,1': 'iPhone 7',
        'iPhone9,3': 'iPhone 7',
        'iPhone9,2': 'iPhone 7 Plus',
        'iPhone9,4': 'iPhone 7 Plus',

        'iPhone10,1': 'iPhone 8',
        'iPhone10,4': 'iPhone 8',
        'iPhone10,2': 'iPhone 8 Plus',
        'iPhone10,5': 'iPhone 8 Plus',
        'iPhone10,3': 'iPhone X',
        'iPhone10,6': 'iPhone X',

        'iPhone11,8': 'iPhone XR',
        'iPhone11,2': 'iPhone XS',

        'iPhone11,6': 'iPhone XS Max',
        'iPhone11,4': 'iPhone XS Max',
    }



def read_data_metric_pool_h5andcsv(file_list,flag_zscore=False,num_core=20,flag_delete_anomaly=False,flag_multilabel=False,mean_x = None,std_x=None,flag_typecoded = True,flag_includeTime=False,flag_191feat = False,flag_select_device_type=False,selected_device_type=None,flag_float32=False,flag_no_pressure=False):
#     print file_list
    len_file_list = len(file_list)
    n_core = int(np.minimum(len_file_list,num_core))
    if n_core>1:
        with poolcontext(processes=n_core) as pool:
            results = pool.map(read_h5andcsv_unpack, [(file_path[0],file_path[1],flag_delete_anomaly,flag_multilabel,flag_typecoded,flag_includeTime,flag_191feat,flag_select_device_type,selected_device_type,flag_no_pressure) for file_path in file_list])
        X,Y = np.vstack([e[0] for e in results]),np.vstack([e[1] for e in results])
    else:
        X,Y = read_h5andcsv(file_list[0][0],file_list[0][1],flag_delete_anomaly,flag_multilabel,flag_typecoded,flag_includeTime,flag_191feat,flag_select_device_type,selected_device_type,flag_no_pressure)
        
   
        
 
    if flag_zscore:
        if flag_includeTime:  
            X[:,2:] = my_zscore_test(X[:,2:],mean_x,std_x)
        else:
            X = my_zscore_test(X,mean_x,std_x)
    if flag_float32:
        X = X.astype(np.float32)

    return  X,Y

def read_data_metric_pool_h5andcsv_v1(file_list,flag_zscore=False,num_core=20,flag_delete_anomaly=False,flag_multilabel=False,mean_x = None,std_x=None,flag_typecoded = True,flag_includeTime=False,flag_191feat = False,flag_select_device_type=False,selected_device_type=None):
#     print file_list
    len_file_list = len(file_list)
    n_core = int(np.minimum(len_file_list,num_core))
    if n_core>1:
        with poolcontext(processes=n_core) as pool:
            results = pool.map(read_csv, [file_path[0] for file_path in file_list])
        mat_str = np.vstack(results)
        with poolcontext(processes=n_core) as pool:
            results = pool.map(read_h5, [file_path[1] for file_path in file_list])
        mat_float = np.vstack(results)
    else:
        mat_str = read_csv(file_list[0][0])
        mat_float = read_h5(file_list[0][1])
        
        
#     print mat_str.shape,mat_float.shape

#     print mat_str[:10]
 
    if flag_multilabel:
        model_type = mat_str[:,1]
        if flag_typecoded:
            dype_values = ID_TO_PHONE_MODEL_MAPPING.values()
            model_use = np.array([e_type in dype_values for e_type in model_type])
        else:
            model_use = np.array([e_type in ID_TO_PHONE_MODEL_MAPPING for e_type in model_type])
        mat_str = mat_str[model_use,:]
        mat_float = mat_float[model_use,:]
        if not flag_typecoded:
            model_type = mat_str[:,1]
            mat_str[:,1] = np.array([ID_TO_PHONE_MODEL_MAPPING[e_type] for e_type in model_type])
        if flag_select_device_type:
            device_type_use = mat_str[:,1] == selected_device_type
            mat_str = mat_str[device_type_use,:]
            mat_float = mat_float[device_type_use,:]
 
        Y = mat_str
    else:
        Y = mat_str[:,0][:,np.newaxis] 
        
    if flag_includeTime:    
        X = mat_float
    else:
        X = mat_float[:,2:]
        
    if flag_191feat:
        X = X[:,:-(974-191)]
    
#     print 'X.shape',X.shape
    
#     print(len(X))  
    if flag_delete_anomaly:
        anomaly = mat_float[:,0]
        X = X[anomaly==0,:]
        Y = Y[anomaly==0,:]
    
#     print(len(X)) 

    nan_idx = np.sum(np.isnan(X),axis=1) == 0
    X = X[nan_idx,:]
    Y = Y[nan_idx,:]
    
#     print(len(X)) 
  
    if flag_zscore:
        if flag_includeTime:  
            X[:,2:] = my_zscore_test(X[:,2:],mean_x,std_x)
        else:
            X = my_zscore_test(X,mean_x,std_x)
 
    
    return  X,Y

def read_data_metric(file_list,flag_zscore=False):
 
    for i_file,file_path in enumerate(file_list):
        df = pd.read_csv(file_path,header=0)   
        mat = df.values
        Y = mat[:,0][:,np.newaxis] 
        X = mat[:,-191:]
 
        nan_idx = np.sum(np.isnan(X),axis=1) == 0
        X = X[nan_idx,:]
        Y = Y[nan_idx,:]
 
        if i_file == 0:
            XX = X
            YY = Y
        else:
            XX = np.vstack([XX,X])
            YY = np.vstack([YY,Y])
            
    if flag_zscore:
        save_root_path = 'extracted_transformer_pickle'
        write_path = 'param_wechat_ds20190120_fold_4_step_100000.p'
        param_dic = load_param(save_root_path,write_path)
        mean_x = param_dic['mean_x']
        std_x = param_dic['std_x']
        XX = my_zscore_test(XX,mean_x,std_x)
 
    return  XX,YY 

def clean_data(train_X_f,y):
    nan_idx = np.sum(np.isnan(train_X_f),axis=1) == 0
    train_X_f = train_X_f[nan_idx,:]
    y =y[nan_idx,:]
    nan_idx = np.sum(np.isinf(train_X_f),axis=1) == 0
    train_X_f = train_X_f[nan_idx,:]
    y =y[nan_idx,:]
    return train_X_f,y

def eval_distBased_trainValid_fixFAR_negDownsample(i,str_input,test_list,batch_size_list_test,Model_gen_hidden_feature_branch,flag_multilabel=False,feature_dim=128,read_method=read_data_metric_pool,flag_zscore=True,nsample=40,mean_x = None,std_x=None,flag_multimodel=False,dic_deviceType2ID=None,flag_typecoded = True,flag_191feat=False,flag_select_device_type=False,selected_device_type=None,flag_test_same_device=True,flag_delete_anomaly=False,flag_float32=False,flat_time_float32=False,flag_gen_mask=False,session=None,mask=None,inputs=None,input_target=None,batch_size=512,reader_vector_class_sort=None,ds='ds20190414',flag_no_pressure=False):
    cur_list_test = 0
    i_prob = 0
    FAR_ALL = []
    FRR_ALL = []
    
    while (cur_list_test+1)*batch_size_list_test<=len(test_list):
        test_list_tmp = test_list[cur_list_test*batch_size_list_test:(cur_list_test+1)*batch_size_list_test]
        cur_list_test = cur_list_test + 1
        test_train_X,test_train_y_a = read_method(test_list_tmp,flag_zscore=flag_zscore,flag_delete_anomaly=flag_delete_anomaly,mean_x = mean_x,std_x=std_x,flag_multilabel=flag_multilabel,flag_typecoded = flag_typecoded,flag_191feat=flag_191feat,flag_includeTime=True,flag_select_device_type=flag_select_device_type,selected_device_type=selected_device_type,flag_float32=False,flag_no_pressure=flag_no_pressure)
        
        timestamp = test_train_X[:,1]
        if flat_time_float32:
            timestamp = timestamp.astype(np.float32)
        test_train_idx,test_test_idx,test_num_sample = split_train_test_fix_nsample_timeordered(test_train_y_a[:,i_prob],timestamp,test_rate = 0.2,nsample=nsample)
        
        test_train_X = test_train_X[:,2:]
        if flag_float32:
            test_train_X = test_train_X.astype(np.float32)
        
        
        
        test_test_X = test_train_X[test_test_idx]
        test_test_y_a = test_train_y_a[test_test_idx]
        test_train_X = test_train_X[test_train_idx]
        test_train_y_a = test_train_y_a[test_train_idx]
        
        if flag_gen_mask:
            
            data_reader_test_train = reader_vector_class_sort.Reader(test_train_X,test_train_y_a, batch_size=batch_size,flag_shuffle=False)  
            data_reader_test_train_rand = reader_vector_class_sort.Reader(test_train_X,test_train_y_a, batch_size=batch_size,flag_shuffle=True)  
            
            input_shape = (test_train_X.shape[1],)
            if cur_list_test==1:
                count, new_mean, M2 = 0,np.zeros(input_shape[0]),np.zeros(input_shape[0])
            for _ in range(test_train_X.shape[0]/batch_size):
                x_batch, y_batch  = data_reader_test_train.iterate_batch()
                x_batch_rand, y_batch_rand  = data_reader_test_train_rand.iterate_batch()
                if x_batch.shape[0]!=batch_size or x_batch_rand.shape[0]!=batch_size:
                    continue
                x_batch = np.vstack([x_batch,x_batch_rand])
                y_batch = np.vstack([y_batch,y_batch_rand])
                y_batch_task = gen_identical_mask(y_batch[:, 0])
                mask_batch = session.run(mask, feed_dict={inputs:x_batch,input_target:y_batch_task})
#                 print 'mask_batch.shape,new_mean.shape,M2.shape',mask_batch.shape,new_mean.shape,M2.shape
                count, new_mean, M2=update_mean_M2_minibatch((count, new_mean, M2), mask_batch)
            data_reader_test_train.close()
            data_reader_test_train_rand.close()
            
            new_std = np.sqrt(M2/float(count-1))
            print np.mean(new_mean),np.mean(new_std)
            
            a = np.sort(new_mean)
            a = a[-1::-1]
            th = a[300]
            selector = np.zeros(new_mean.shape)
            selector[new_mean>th]=1.
            selector[:191]=1.
            
            print np.sum(selector==1.)
            
#             new_mean = new_mean[np.newaxis,:]
            test_train_X = test_train_X*selector[np.newaxis,:]
            test_test_X = test_test_X*selector[np.newaxis,:]
            
            

        if flag_multimodel:
            test_train_y_a[:, 1] = label2uniqueID_sub_test(test_train_y_a[:, 1],dic_deviceType2ID)[:,0]
            train_X_f=Model_gen_hidden_feature_branch.predict([test_train_X,to_categorical(test_train_y_a[:, 1].astype(int), len(dic_deviceType2ID))])
            test_test_y_a[:, 1] = label2uniqueID_sub_test(test_test_y_a[:, 1],dic_deviceType2ID)[:,0]
            test_X_f=Model_gen_hidden_feature_branch.predict([test_test_X,to_categorical(test_test_y_a[:, 1].astype(int), len(dic_deviceType2ID))])   
        else:
            train_X_f=Model_gen_hidden_feature_branch.predict(test_train_X)
            test_X_f=Model_gen_hidden_feature_branch.predict(test_test_X)
        
#         print 'train_X_f.shape',train_X_f.shape
        train_X_f,test_train_y_a = clean_data(train_X_f,test_train_y_a)
        test_X_f,test_test_y_a = clean_data(test_X_f,test_test_y_a)
#         print 'train_X_f.shape',train_X_f.shape
        
        if flag_multilabel:
#             print feature_dim
            train_X_f = train_X_f[:,:feature_dim]
            test_X_f = test_X_f[:,:feature_dim]
            
            if flag_test_same_device:
                _, _,far,frr = eval_distBased_valid_fixFAR_nopool_negDownsample_sameDeviceType(train_X_f,test_train_y_a,test_X_f,test_test_y_a,0,'',flag_print=False)
            else:
                _, _,far,frr = eval_distBased_valid_fixFAR_nopool_negDownsample(train_X_f,test_train_y_a[:, i_prob],test_X_f,test_test_y_a[:, i_prob],0,'',flag_print=False)
                
        else:
            _, _,far,frr = eval_distBased_valid_fixFAR_nopool_negDownsample(train_X_f,test_train_y_a[:, i_prob],test_X_f,test_test_y_a[:, i_prob],0,'',flag_print=False)

        if cur_list_test == 1:
            mean_far,mean_frr,cont_far,cont_frr=np.mean(far),np.mean(frr),len(far),len(frr)
        else:
            sum_cont_far,sum_cont_frr = cont_far+len(far),cont_frr+len(frr)
            mean_far,mean_frr = (mean_far*cont_far + np.sum(far))/float(sum_cont_far),(mean_frr*cont_frr + np.sum(frr))/float(sum_cont_frr)
            cont_far,cont_frr = sum_cont_far,sum_cont_frr
    if flag_gen_mask:        
        f = h5py.File('/data/ceph/notebooks/joshua/sensor_touch_wechat/data/mean_std_mask_wechat_%s.h5' % ds,'w')
        f['mean_mask'] = new_mean
        f['std_mask'] = new_std
        f.close()
            
    print('%s: step %d, FAR %g, FRR %g, n_eval_far %d, n_eval_frr %d' % (str_input, i, mean_far,mean_frr,cont_far,cont_frr)) 
    return 1-mean_frr


def eval_angleBased_trainValid_fixFAR_negDownsample(i,str_input,test_list,batch_size_list_test,Model_gen_hidden_feature_branch,flag_multilabel=False,feature_dim=128,read_method=read_data_metric_pool,flag_zscore=True,nsample=40,mean_x = None,std_x=None,flag_multimodel=False,dic_deviceType2ID=None,flag_typecoded = True,flag_191feat=False,flag_select_device_type=False,selected_device_type=None,flag_test_same_device=True):
    cur_list_test = 0
    i_prob = 0
    FAR_ALL = []
    FRR_ALL = []
    
    while (cur_list_test+1)*batch_size_list_test<=len(test_list):
        test_list_tmp = test_list[cur_list_test*batch_size_list_test:(cur_list_test+1)*batch_size_list_test]
        cur_list_test = cur_list_test + 1
        test_train_X,test_train_y_a = read_method(test_list_tmp,flag_zscore=flag_zscore,flag_delete_anomaly=False,mean_x = mean_x,std_x=std_x,flag_multilabel=flag_multilabel,flag_typecoded = flag_typecoded,flag_191feat=flag_191feat,flag_includeTime=True,flag_select_device_type=flag_select_device_type,selected_device_type=selected_device_type)
        
        timestamp = test_train_X[:,1]
        test_train_idx,test_test_idx,test_num_sample = split_train_test_fix_nsample_timeordered(test_train_y_a[:,i_prob],timestamp,test_rate = 0.2,nsample=nsample)
        
        test_train_X = test_train_X[:,2:]
        
        test_test_X = test_train_X[test_test_idx]
        test_test_y_a = test_train_y_a[test_test_idx]
        test_train_X = test_train_X[test_train_idx]
        test_train_y_a = test_train_y_a[test_train_idx]

        if flag_multimodel:
            test_train_y_a[:, 1] = label2uniqueID_sub_test(test_train_y_a[:, 1],dic_deviceType2ID)[:,0]
            train_X_f=Model_gen_hidden_feature_branch.predict([test_train_X,to_categorical(test_train_y_a[:, 1].astype(int), len(dic_deviceType2ID))])
            test_test_y_a[:, 1] = label2uniqueID_sub_test(test_test_y_a[:, 1],dic_deviceType2ID)[:,0]
            test_X_f=Model_gen_hidden_feature_branch.predict([test_test_X,to_categorical(test_test_y_a[:, 1].astype(int), len(dic_deviceType2ID))])   
        else:
            train_X_f=Model_gen_hidden_feature_branch.predict(test_train_X)
            test_X_f=Model_gen_hidden_feature_branch.predict(test_test_X)
        
#         print 'train_X_f.shape',train_X_f.shape
        train_X_f,test_train_y_a = clean_data(train_X_f,test_train_y_a)
        test_X_f,test_test_y_a = clean_data(test_X_f,test_test_y_a)
#         print 'train_X_f.shape',train_X_f.shape
        
        if flag_multilabel:
#             print feature_dim
            train_X_f = train_X_f[:,:feature_dim]
            test_X_f = test_X_f[:,:feature_dim]
            
            if flag_test_same_device:
                _, _,far,frr = eval_angleBased_valid_fixFAR_nopool_negDownsample_sameDeviceType(train_X_f,test_train_y_a,test_X_f,test_test_y_a,0,'',flag_print=False)
            else:
                _, _,far,frr = eval_distBased_valid_fixFAR_nopool_negDownsample(train_X_f,test_train_y_a[:, i_prob],test_X_f,test_test_y_a[:, i_prob],0,'',flag_print=False)
                
        else:
            _, _,far,frr = eval_distBased_valid_fixFAR_nopool_negDownsample(train_X_f,test_train_y_a[:, i_prob],test_X_f,test_test_y_a[:, i_prob],0,'',flag_print=False)

        if cur_list_test == 1:
            mean_far,mean_frr,cont_far,cont_frr=np.mean(far),np.mean(frr),len(far),len(frr)
        else:
            sum_cont_far,sum_cont_frr = cont_far+len(far),cont_frr+len(frr)
            mean_far,mean_frr = (mean_far*cont_far + np.sum(far))/float(sum_cont_far),(mean_frr*cont_frr + np.sum(frr))/float(sum_cont_frr)
            cont_far,cont_frr = sum_cont_far,sum_cont_frr
            

    print('%s: step %d, FAR %g, FRR %g, n_eval_far %d, n_eval_frr %d' % (str_input, i, mean_far,mean_frr,cont_far,cont_frr)) 
    return 1-mean_frr

def sub_fun_offline(i,clf,train_X,label_train,test_X,label_test,test_X_attack,label_test_attack):
#     print i
    clf.fit(train_X, label_train)
    prob = clf.predict_proba(test_X)[:,1]
    prob_attack = clf.predict_proba(test_X_attack)[:,1]

    idx = i
    
#     dic_th = argmin_mean_FRR_st_FAR(label_test, prob,exp_far=1e-5,pen_far=1e20)
#     sorted_prob = np.sort(prob[label_test==1])
#     idx_th = int(0.05*len(sorted_prob))
#     dic_th = sorted_prob[idx_th]
#     dic_th = argmin_mean_FAR_FRR(label_test, prob)
#     print len(sorted_prob),idx_th/float(len(sorted_prob))
    
    dic_th = argmin_fixFRR(label_test, prob,exp_frr=0.05)
    label_out = prob > dic_th
    label_out_attack = prob_attack > dic_th
    FAR_ALL = FAR_score(label_test,label_out)
    FAR_ALL_attack = 1 - FRR_score(label_test_attack,label_out_attack)
    FRR_ALL = FRR_score(label_test,label_out)
    ACC_ALL = np.mean(label_test == label_out)
    sum_1_label_test = np.sum(label_test)
    sum_1_label_out = np.sum(label_out*label_test)
    sum_1_label_out_attack = np.sum(label_out_attack*label_test_attack)
    sum_1_label_test_attack = np.sum(label_test_attack)
    

#     print('%d    : FAR %g, FAR_attack %g, FRR %g, acc %g' % (i, FAR_ALL,FAR_ALL_attack,FRR_ALL,ACC_ALL))
        
    return (FAR_ALL,FRR_ALL,ACC_ALL,FAR_ALL_attack,
            sum_1_label_test,sum_1_label_out,sum_1_label_out_attack,sum_1_label_test_attack)

def sub_fun_offline_unpack(args):
    return sub_fun_offline(*args)


def eval_offline(read_path,use_wechet_data = True,extract_feature = True,use_default_model=False,flag_include_zscore=False):
    
    save_root_path = 'extracted_transformer_pickle'
    write_path = 'param_unspvGAN_974feat_cluster2_VAT_wechat_ds20190414_recall_9397.p'
    param_dic = load_param(save_root_path,write_path)
    mean_x = param_dic['mean_x']
    std_x = param_dic['std_x']
 
    
    mean_x = mean_x[:191]
    std_x = std_x[:191]
    
    save_root_path = 'saved_model_transformer'
#     read_path = 'transfer_unspvGAN_norand_h1024_dep6_191feat_wechat_ds20190414_recall_9243.h5'
    
    if flag_include_zscore:
        model_tran = load_model(os.path.join(save_root_path,read_path),custom_objects={'ZscoreLayer': ZscoreLayer})
        param_dic = param2numpy_transfer_include_zscore(model_tran)
    else:
        model_tran = load_model(os.path.join(save_root_path,read_path))
        param_dic = param2numpy_transfer(model_tran,mean_x=mean_x,std_x=std_x)
    save_root_path = 'extracted_transformer_pickle'
    from scanf import scanf
    str_name = scanf('%s.h5',read_path)
    write_path = 'param_%s.p' % str_name[0]
    save_param(save_root_path,write_path,param_dic)
 
    data_path = '/data/ceph/notebooks/joshua/transfer/data/offline_data/'
    
    
    label_train = np.load(os.path.join(data_path,'offline_label_train.npy'))
    label_test = np.load(os.path.join(data_path,'offline_label_test.npy'))
    label_test_attack = np.load(os.path.join(data_path,'offline_label_test_attack.npy')) 
    train_X = np.load(os.path.join(data_path,'offline_train_X.npy'))
    test_X = np.load(os.path.join(data_path,'offline_test_X.npy'))
    test_X_attack = np.load(os.path.join(data_path,'offline_test_X_attack.npy'))
    train_X_aux = np.load(os.path.join(data_path,'offline_train_X_aux.npy'))
    
    if extract_feature:

        save_root_path = 'extracted_transformer_pickle'
        if use_default_model:
            write_path = 'param_unspvGAN_191feat_cluster2_wechat_ds20190414_recall_9165.p'

        param_dic = load_param(save_root_path,write_path)


        train_X = transformer(train_X,param_dic)
        test_X = transformer(test_X,param_dic)
        test_X_attack = transformer(test_X_attack,param_dic)

        if use_wechet_data:
            train_X_aux = transformer(train_X_aux,param_dic)
 
    if use_wechet_data:
        train_X = np.vstack([train_X,train_X_aux])
        label_train = np.vstack([label_train,np.zeros((train_X_aux.shape[0],label_train.shape[1]))])
    
 
    idx = 0
    model_type = 'lr'
    if model_type == 'lr':
#         c_param = 2**(-8)
        c_param = 1.
        clf = LogisticRegression(C=c_param,class_weight='balanced',solver='liblinear')
    
    if model_type == 'xgb':
         
        clf = XGBClassifier(max_depth=8, learning_rate=0.1, n_estimators=360)
    
    if model_type == 'rf':
        n_tree = 360
        clf = RandomForestClassifier(n_estimators=n_tree,class_weight='balanced', random_state=0)
    
          
    num_core = 20
    with poolcontext(processes=num_core) as pool:
        results = pool.map(sub_fun_offline_unpack, [(i,clf,train_X,label_train[:,i],test_X,label_test[:,i],test_X_attack,label_test_attack[:,i]) for i in range(label_train.shape[1])])
    
    results = np.array(results)
    
    FAR_ALL = results[:,0]
    FRR_ALL = results[:,1]
    ACC_ALL = results[:,2]
    FAR_ALL_attack =  results[:,3]
    
    sum_1_label_test = results[:,4]
    sum_1_label_out = results[:,5]
    sum_1_label_out_attack = results[:,6]
    sum_1_label_test_attack = results[:,7]
    
    print('FAR %g, FRR %g, acc %g, FAR_ATTACK %g' % (np.mean(FAR_ALL),np.mean(FRR_ALL[np.invert(np.isnan(FRR_ALL))]),np.mean(ACC_ALL),np.mean(FAR_ALL_attack[np.invert(np.isnan(FAR_ALL_attack))])))
    
    
    recall_macro = 1. - np.mean(FRR_ALL[np.invert(np.isnan(FRR_ALL))])
    attack_macro = np.mean(FAR_ALL_attack[np.invert(np.isnan(FAR_ALL_attack))])
    print('with model:')
    print('macro: recall %g,\t\t\tattack %g' %(recall_macro,attack_macro))
    print('micro: recall %g(%d/%d),\tattack %g(%d/%d)' %(np.sum(sum_1_label_out)/np.sum(sum_1_label_test),np.sum(sum_1_label_out),np.sum(sum_1_label_test),
                                                          np.sum(sum_1_label_out_attack)/np.sum(sum_1_label_test_attack),
                                                         np.sum(sum_1_label_out_attack),np.sum(sum_1_label_test_attack)))
    print('without model:')
    eval_distBased_testattack(train_X,label_train,
                              test_X,label_test,test_X_attack,label_test_attack,0,'',argmin_fun=argmin_fixFRR,exp_frr=0.2)
 
    return recall_macro,attack_macro
 

def sub_fun_dist_test(i_user,test_X,Y_test,train_X,dic_th):
    idx = i_user
    
    dist = mat_dist(test_X,train_X)
    prob = - np.min(dist,axis=1)
 
    label_out = prob > dic_th 
    FAR_ALL = FAR_score(Y_test,label_out)
    return FAR_ALL 

def sub_fun_dist_test_unpack(args):
    return sub_fun_dist_test(*args)

def eval_distBased_test_attack_pool(train_X,train_Y,test_X,i,str_input,dic_th,num_core=20):
 
    n = test_X.shape[0]

    
    final_labels = np.unique(train_Y)
 
    
    n_user_train = len(final_labels)

     
        
    with poolcontext(processes=num_core) as pool:
        results = pool.map(sub_fun_dist_test_unpack, [(i_user,test_X,np.zeros(n),train_X[train_Y==i_user,:],dic_th[i_user]) for i_user in final_labels])
      
    results = np.array(results)
    FAR_ALL = results 
    


    print('%s: step %d, FAR %g, n_eval %d' % (str_input, i,   np.mean(FAR_ALL),n_user_train))
     
 

    return

def eval_distBased_test_attack_nopool(train_X,train_Y,test_X,i,str_input,dic_th,num_core=20):
 
    n = test_X.shape[0]

    
    final_labels = np.unique(train_Y)
 
    
    n_user_train = len(final_labels)
    
    FAR_ALL = np.zeros(n_user_train)
    start_time = time_simple.clock() 
    for i_user in final_labels:
        if i_user % 1000 == 0:
            print i_user
            end_time = time_simple.clock()
            print 'testing time: %.3f' % (end_time - start_time)
            start_time = time_simple.clock() 
            
        FAR_ALL[i_user]=sub_fun_dist_test(i_user,test_X,np.zeros(n),train_X[train_Y==i_user,:],dic_th[i_user])

      


    print('%s: step %d, FAR %g, n_eval %d' % (str_input, i,   np.mean(FAR_ALL),n_user_train))
     
 

    return

def sub_fun_dist_test_real_attack(i_user,test_X,Y_test,train_X,dic_th):
    idx = i_user
#     print test_X.shape,Y_test.shape
    dist = mat_dist(test_X,train_X)
    prob = - np.min(dist,axis=1)
 
    label_out = prob > dic_th 
    FRR_ALL = FRR_score(Y_test,label_out)
    return FRR_ALL 

def eval_distBased_test_attack_nopool_real_attack(train_X,train_Y,test_X,test_Y,i,str_input,dic_th,num_core=20):
 
    n = test_X.shape[0]

    
    final_labels = np.unique(test_Y)
 
    
    n_user_train = len(final_labels)
    
    FRR_ALL = np.zeros(n_user_train)
    start_time = time_simple.clock() 
    for i_order,i_user in enumerate(final_labels):
#         if i_user % 1000 == 0:
#             print i_user
#             end_time = time_simple.clock()
#             print 'testing time: %.3f' % (end_time - start_time)
#             start_time = time_simple.clock() 
            
        FRR_ALL[i_order]=sub_fun_dist_test_real_attack(i_user,test_X,(test_Y==i_user).astype(int),train_X[train_Y==i_user,:],dic_th[i_user])

      


    print('%s: step %d, FAR %g, n_eval %d' % (str_input, i,   1-np.mean(FRR_ALL),n_user_train))
     
 

    return

def to_categorical(y,nb_classes):
    return np.eye(nb_classes)[y]



def mat_angle(qf,gf):
    return np.dot(qf,gf.T)


def sub_fun_angle(i_user,test_X,Y_test,train_X):
    
    dist = mat_angle(test_X,train_X)
    prob = np.max(dist,axis=1)
    
    
    idx = i_user
    dic_th = argmin_mean_FRR_st_FAR(Y_test, prob)

    label_out = prob > dic_th
    FAR_ALL = FAR_score(Y_test,label_out)
    FRR_ALL = FRR_score(Y_test,label_out)
    ACC_ALL = np.mean(Y_test == label_out)
    return FAR_ALL,FRR_ALL,ACC_ALL,dic_th

def sub_fun_angle_unpack(args):
    return sub_fun_angle(*args)

@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()
    
def eval_angleBased_valid_fixFAR_pool(train_X,train_Y,test_X,test_Y,i,str_input,num_core=20):
     
        
    dic_th = {}
   
    test_labels = np.unique(test_Y)
    train_labels = np.unique(train_Y)
    final_labels = np.intersect1d(train_labels,test_labels)
    
    n_user_train = len(final_labels)
     
#     print 'np.any(np.isnan(train_X)),np.any(np.isnan(test_X))',np.any(np.isnan(train_X)),np.any(np.isnan(test_X))
 
    with poolcontext(processes=num_core) as pool:
        results = pool.map(sub_fun_angle_unpack, [(i_user,test_X,(test_Y==i_user).astype(int),train_X[train_Y==i_user,:]) for i_user in final_labels])
    
    results = np.array(results)
    
    FAR_ALL = results[:,0]
    FRR_ALL = results[:,1]
    ACC_ALL = results[:,2]
    dic_th =  results[:,3]
    
         


    print('%s: step %d, FAR %g, FRR %g, acc_avg %g, n_eval %d' % (str_input, i, np.mean(FAR_ALL),np.mean(FRR_ALL),np.mean(ACC_ALL),n_user_train))
#     print('%s: step %d, [%g, %g, %g, %g, %g]' % (str_input, i,  np.mean(auc_all), acc,np.mean(ACC_ALL),np.mean(FAR_ALL),np.mean(FRR_ALL)))
   
  
    
    return dic_th,1-np.mean(FRR_ALL)


def eval_angleBased_valid_fixFAR_nopool(train_X,train_Y,test_X,test_Y,i,str_input,num_core=20):
     
        
    dic_th = {}
   
    test_labels = np.unique(test_Y)
    train_labels = np.unique(train_Y)
    final_labels = np.intersect1d(train_labels,test_labels)
    
    n_user_train = len(final_labels)
     
    FAR_ALL = np.zeros(n_user_train)
    FRR_ALL = np.zeros(n_user_train)
    ACC_ALL = np.zeros(n_user_train)
    dic_th  = np.zeros(n_user_train)
    start_time = time_simple.clock() 
    for i_user in final_labels:
        if i_user % 1000 == 0:
            print i_user
            end_time = time_simple.clock()
            print 'testing time: %.3f' % (end_time - start_time)
            start_time = time_simple.clock() 
            
        FAR_ALL[i_user],FRR_ALL[i_user],ACC_ALL[i_user],dic_th[i_user] = sub_fun_angle(i_user,test_X,(test_Y==i_user).astype(int),train_X[train_Y==i_user,:])
         


    print('%s: step %d, FAR %g, FRR %g, acc_avg %g, n_eval %d' % (str_input, i, np.mean(FAR_ALL),np.mean(FRR_ALL),np.mean(ACC_ALL),n_user_train))
#     print('%s: step %d, [%g, %g, %g, %g, %g]' % (str_input, i,  np.mean(auc_all), acc,np.mean(ACC_ALL),np.mean(FAR_ALL),np.mean(FRR_ALL)))
   
  
    
    return dic_th,1-np.mean(FRR_ALL)

def sub_fun_angle_test(i_user,test_X,Y_test,train_X,dic_th):
    idx = i_user
    
    dist = mat_angle(test_X,train_X)
    prob = np.max(dist,axis=1)
 
    label_out = prob > dic_th 
    FAR_ALL = FAR_score(Y_test,label_out)
    return FAR_ALL 

def sub_fun_angle_test_unpack(args):
    return sub_fun_angle_test(*args)

def eval_angleBased_test_attack_pool(train_X,train_Y,test_X,i,str_input,dic_th,num_core=20):
 
   
    n = test_X.shape[0]
  
    final_labels = np.unique(train_Y)
    
    
    n_user_train = len(final_labels)

     
        
    with poolcontext(processes=num_core) as pool:
        results = pool.map(sub_fun_angle_test_unpack, [(i_user,test_X,np.zeros(n),train_X[train_Y==i_user,:],dic_th[i_user]) for i_user in final_labels])
      
    results = np.array(results)
    FAR_ALL = results 
    


    print('%s: step %d, FAR %g, n_eval %d' % (str_input, i,   np.mean(FAR_ALL),n_user_train))
     
 

    return

def eval_angleBased_test_attack_nopool(train_X,train_Y,test_X,i,str_input,dic_th,num_core=20):
 
   
    n = test_X.shape[0]
  
    final_labels = np.unique(train_Y)
    
    
    n_user_train = len(final_labels)

     
        
    n_user_train = len(final_labels)
    
    FAR_ALL = np.zeros(n_user_train)
    start_time = time_simple.clock() 
    for i_user in final_labels:
        if i_user % 1000 == 0:
            print i_user
            end_time = time_simple.clock()
            print 'testing time: %.3f' % (end_time - start_time)
            start_time = time_simple.clock() 
            
        FAR_ALL[i_user]=sub_fun_angle_test(i_user,test_X,np.zeros(n),train_X[train_Y==i_user,:],dic_th[i_user])


    


    print('%s: step %d, FAR %g, n_eval %d' % (str_input, i,   np.mean(FAR_ALL),n_user_train))
     
 

    return

def get_prior(y):
    uni_set = np.unique(y)
    n = len(uni_set)
    p = np.zeros(n)
    for i,e in enumerate(uni_set):
        p[i] = np.sum(y==e)
    return p/np.sum(p)

def gen_idx_fix_n_sample_per_class(y_train,n_sample_per_class):
         
    n = len(y_train)

    idx = np.arange(n)
    class_set = np.unique(y_train)
    n_class = len(class_set)
#         print 'n_class',n_class
    idx_uni_class = np.arange(n_class)


#         np.random.shuffle(class_set)
    idx_ret = []
    for e_class in class_set:
        idx_e = idx[y_train==e_class]
        idx_e = np.random.choice(idx_e, size=n_sample_per_class, replace=False)
        idx_ret.extend(idx_e)

    return idx_ret

def split_train_test_fix_nsample_timeordered(y,t,test_rate = 0.2,nsample=40):
    n = y.shape[0]
    class_set,class_counts = np.unique(y,return_counts=True)
    class_set = class_set[class_counts>=nsample] 
    idx = np.arange(n)
    test_idx = []
    train_idx = []
    num_sample = np.zeros(len(class_set))
    for i,e in enumerate(class_set):
        
        idx_e = idx[y == e]
        t_e = t[idx_e]
        idx_t = np.argsort(t_e)
        idx_e = idx_e[idx_t]
        num_sample_i = len(idx_e)
        num_sample[i] = num_sample_i
        if num_sample_i<nsample:
            continue
        th = int(np.ceil(num_sample_i*test_rate))
        test_idx.extend(idx_e[:th])
        train_idx.extend(idx_e[th:])
        
         
 
    return train_idx,test_idx,num_sample

def split_train_test_fix_nsample(y,test_rate = 0.2,nsample=40):
    n = y.shape[0]
    class_set,class_counts = np.unique(y,return_counts=True)
    class_set = class_set[class_counts>=nsample] 
    idx = np.arange(n)
    test_idx = []
    train_idx = []
    num_sample = np.zeros(len(class_set))
    for i,e in enumerate(class_set):
        
        idx_e = idx[y == e]
        np.random.shuffle(idx_e)
        num_sample_i = len(idx_e)
        num_sample[i] = num_sample_i
        if num_sample_i<nsample:
            continue
        th = int(np.ceil(num_sample_i*test_rate))
        test_idx.extend(idx_e[:th])
        train_idx.extend(idx_e[th:])
        
         
 
    return train_idx,test_idx,num_sample


def update_mean_M2_minibatch(existingAggregate, newValue):
    (count, mean, M2) = existingAggregate
    b = len(newValue)
    s = np.sum(newValue,0) 
    count += b 
    delta = s - b*mean
    new_mean = mean + delta / count
    M2 += np.sum((newValue-mean)**2,0) - count * (mean - new_mean)**2

    return (count, new_mean, M2)

from time import sleep
import sys
def compute_fun():
    a = np.random.random((3,3))
    for _ in range(100):
        a = np.dot(a,a)
        sys.stdout.write('.')
    sleep(1.)
    print '.'

    return

