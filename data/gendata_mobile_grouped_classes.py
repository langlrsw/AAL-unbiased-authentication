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
 
def gen_raw_data():
    data_name = 'sensor_touch'
    print 'reading %s data' % data_name
    
    sys_root_path = "data"

    write_path = os.path.join(sys_root_path,'device_transfer.h5')
    f = h5py.File(write_path, 'r')
    X,Y = np.array(f['X']),np.array(f['Y'])
    f.close()
    print X.shape,Y.shape

    user_ios = np.unique(Y[Y[:,1]==0,0])  
    user_and = np.unique(Y[Y[:,1]==1,0])
    user_intersect = np.intersect1d(user_ios,user_and)  
    user_ios_uni = np.setdiff1d(user_ios,user_intersect)
    user_and_uni = np.setdiff1d(user_and,user_intersect)

    print user_ios
    print user_and
    print user_intersect
    print user_ios_uni
    print user_and_uni
    print np.median(user_intersect),np.mean(user_intersect<np.median(user_intersect)) 

    flag_in_intersect = np.array([e in user_intersect for e in Y[:,0]])
    flag_less  = np.array([e < np.median(user_intersect) for e in Y[:,0]]) 
    flag_greater  = np.array([e >= np.median(user_intersect) for e in Y[:,0]])
 
    idx_test =  (flag_in_intersect & (Y[:,1]==0) & flag_less) |  (flag_in_intersect & (Y[:,1]==1) & flag_greater)
    idx_train = np.array([not e for e in idx_test])

    train_X, train_y_a, test_X, test_y_a = X[idx_train,:],Y[idx_train,:],X[idx_test,:],Y[idx_test,:]

    ATTR_NUM = train_y_a.shape[1]  
    CLASS_NUM = [len(np.unique(train_y_a[:,i])) for i in range(ATTR_NUM)]  
    #shuffle training data
    indices = np.arange(train_X.shape[0])
    np.random.shuffle(indices)
    train_X = train_X[indices]
    train_y_a = train_y_a[indices]
    
    input_shape = (train_X.shape[1],)  
 
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

if __name__ == "__main__":
    train_X, train_y_a, test_X, test_y_a, train_y_c, test_y_c, ATTR_NUM, CLASS_NUM, input_shape,data_name,lambda_mat,prior_list =gen_raw_data() 

