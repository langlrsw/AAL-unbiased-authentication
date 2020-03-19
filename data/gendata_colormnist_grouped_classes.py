# -*- coding: utf-8 -*-

import numpy as np
import h5py
import os.path
from scipy.io import loadmat
import pandas as pd
  
def label2uniqueID_sub(y):
    dic ={}
    cont = 0
    for i,e in enumerate(y):
        if dic.get(e,-1)==-1:
            dic[e] = cont
            cont = cont + 1
        y[i] = dic[e]
 
    return y

def label2uniqueID(Y):
    
    for i in range(Y.shape[1]):
        y = label2uniqueID_sub(Y[:,i])[:,np.newaxis]
        if i == 0:
            new_Y = y
        else:
            new_Y = np.hstack([new_Y,y])
 
    return new_Y

def label2uniqueID_sub_test(y,dic):
 
    for i,e in enumerate(y):
        y[i] = dic[e]
 
    return y

def label2uniqueID_test(Y,dic_list):
    
    for i in range(Y.shape[1]):
        y = label2uniqueID_sub_test(Y[:,i],dic_list[i])[:,np.newaxis]
        if i == 0:
            new_Y = y
        else:
            new_Y = np.hstack([new_Y,y])
 
    return new_Y

def label2uniqueID_sub_train(y):
    dic ={}
    cont = 0
    for i,e in enumerate(y):
        if dic.get(e,-1)==-1:
            dic[e] = cont
            cont = cont + 1
        y[i] = dic[e]
 
    return y[:,np.newaxis],dic

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
 
def gen_raw_data():
    data_name = 'colored_mnist'
    print 'reading %s data' % data_name
    
    with h5py.File(os.path.join('data', "colored_mnist.h5")) as f:
        train_X, train_y_a, test_X, test_y_a = f["tr_x"][:], f["tr_y"][:], f["te_x"][:], f["te_y"][:]

    color_id_1 = 6  
    color_id_2 = 3  

    a = train_X[(train_y_a[:,0]<=4) & (train_y_a[:,1] ==color_id_1),:,:,:] 
    b = train_X[(train_y_a[:,0]>4) & (train_y_a[:,1] ==color_id_2),:,:,:]
    train_X = np.vstack([a,b])
    a = train_y_a[(train_y_a[:,0]<=4) & (train_y_a[:,1] ==color_id_1),:]
    b = train_y_a[(train_y_a[:,0]>4) & (train_y_a[:,1] ==color_id_2),:]
    train_y_a = np.vstack([a,b])

    a = test_X[(test_y_a[:,0]>4) & (test_y_a[:,1] ==color_id_1),:,:,:]
    b = test_X[(test_y_a[:,0]<=4) & (test_y_a[:,1] ==color_id_2),:,:,:]
    test_X = np.vstack([a,b])
    a = test_y_a[(test_y_a[:,0]>4) & (test_y_a[:,1] ==color_id_1),:]
    b = test_y_a[(test_y_a[:,0]<=4) & (test_y_a[:,1] ==color_id_2),:]
    test_y_a = np.vstack([a,b])
    
    train_y_a,dic_list = label2uniqueID_train(train_y_a)
    test_y_a = label2uniqueID_test(test_y_a,dic_list)
    
    print 'label2idx:',dic_list
     
    train_X = train_X.astype(np.float32) / 255
    test_X = test_X.astype(np.float32) / 255    

    ATTR_NUM = train_y_a.shape[1]  
    CLASS_NUM = [len(np.unique(train_y_a[:,i])) for i in range(ATTR_NUM)]  
    #shuffle training data
    indices = np.arange(train_X.shape[0])
    np.random.shuffle(indices)
    train_X = train_X[indices]
    train_y_a = train_y_a[indices]
     
    input_shape = (28, 28, 3)  
 
    coef = [1]
    for i in range(ATTR_NUM):
        if i == ATTR_NUM - 1:
            break
        coef.append(coef[i]*CLASS_NUM[i])
        
    coef = np.array(coef)[:,np.newaxis]

    train_y_c = train_y_a.dot(coef)  
    test_y_c = test_y_a.dot(coef)

    prior_list = []
    for i in range(ATTR_NUM):
        uniset = np.unique(train_y_a[:,i])
        prior_vec = np.zeros(len(uniset))
        for j,e in enumerate(uniset):
            prior_vec[j] = np.mean(train_y_a[:,i] == e)
        prior_list.append(prior_vec)
        
    print prior_list
    
    lambda_mat = 1 - np.eye(3)
    
    return train_X, train_y_a, test_X, test_y_a, train_y_c, test_y_c, ATTR_NUM, CLASS_NUM, input_shape,data_name,lambda_mat,prior_list



