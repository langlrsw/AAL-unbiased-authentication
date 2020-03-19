# -*- coding: utf-8 -*-

import numpy as np
import h5py
import os.path
from scipy.io import loadmat
import pandas as pd
  
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
    for i in range(Y.shape[1]):
        y,dic = label2uniqueID_sub(Y[:,i])
        if i == 0:
            new_Y = y
        else:
            new_Y = np.hstack([new_Y,y])
        dic_list.append(dic)
 
    return new_Y,dic_list

def label2uniqueID_sub_test(y,dic):
   
    y_new = np.zeros(y.shape).astype(int)
    uni_set = np.unique(y)
    for e in uni_set:
        y_new[y==e]=dic.get(e,-1)

    return y_new[:,np.newaxis] 

def label2uniqueID_test(Y,dic_list):
    
    for i in range(Y.shape[1]):
        y= label2uniqueID_sub_test(Y[:,i],dic_list[i]) 
        if i == 0:
            new_Y = y
        else:
            new_Y = np.hstack([new_Y,y])
 
    return new_Y

def label2uniqueID_sub_train(y):
    dic ={}
    uni_set = np.unique(y)
    y_new = np.zeros(y.shape).astype(int)
    for i,e in enumerate(uni_set):
        y_new[y==e]=i
        dic[e] = i
 
    return y_new[:,np.newaxis],dic

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

 
