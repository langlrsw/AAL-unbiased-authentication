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
 
def gen_raw_data(attr_name_list = ['Eyeglasses']):
    data_name = 'celebA'
    print 'reading %s data' % data_name
    print attr_name_list
    
    path = 'data'
    file_path = os.path.join(path,'attr_independence_mat.xlsx')
    df_mat = pd.read_excel(file_path)
    
    sys_root_path = 'data'
    img_folder = 'img_align_celeba'
    attr_file_name = 'celebA.csv'
    
    df = pd.read_csv(os.path.join(sys_root_path,attr_file_name),index_col='pic')
 

    column_names =df.keys()
    
    head_list = ['person']
    head_list.extend(attr_name_list)
     
    final_list = head_list

    df = df.loc[:,final_list]
    df_mat = df_mat.loc[final_list,final_list]

     
    read_path = os.path.join(sys_root_path,'celeba_img_align_5p_size64.h5')
    f = h5py.File(read_path, 'r')
    X = np.array(f['X'])
    f.close()
 
    Y = df.values
    Y,dic_list = label2uniqueID(Y)
    Y = Y[:-1,:]
     
    
    lambda_mat = df_mat.values
   
    print X.shape,Y.shape,lambda_mat.shape

    user_ios = np.unique(Y[Y[:,1]==0,0])  
    user_and = np.unique(Y[Y[:,1]==1,0])
    user_intersect = np.intersect1d(user_ios,user_and)  
    user_ios_uni = np.setdiff1d(user_ios,user_intersect)
    user_and_uni = np.setdiff1d(user_and,user_intersect)
    
    flag_in_intersect = np.array([e in user_intersect for e in Y[:,0]])
    
    X = X[flag_in_intersect,:,:,:]
    Y = Y[flag_in_intersect,:]
    
    user_id_set = np.unique(Y[:,0])
    
    num_pic = np.array([np.sum(Y[:,0]==user_id) for user_id in user_id_set])
    num_pic_0p5_minus_ratio = np.array([np.abs(0.5-float(np.sum(Y[Y[:,1]==0,0]==user_id))/np.sum(Y[:,0]==user_id)) for user_id in user_id_set])
    th_num_pic = 20
    th_num_pic_0p5_minus_ratio = 0.2
    user_id_set = np.array(user_id_set)
    user_id_set = user_id_set[(num_pic>=th_num_pic) & (num_pic_0p5_minus_ratio<=th_num_pic_0p5_minus_ratio)]
    
    th_num_person = 500
    
    if len(user_id_set) > th_num_person:
        np.random.shuffle(user_id_set)
        user_id_set = user_id_set[:th_num_person]
        
    
    flag_select = np.array([e in user_id_set for e in Y[:,0]])
    
    X = X[flag_select,:,:,:]
    Y = Y[flag_select,:]
    
    Y,dic_list = label2uniqueID(Y)
    
    print 'data ready'
    
    print X.shape,Y.shape
    
#     user_id_set = [np.unique(Y[:,i]) for i in range(Y.shape[1])]
#     print user_id_set
    
    user_id_set = np.unique(Y[:,0])
    
    num_pic = np.array([np.sum(Y[:,0]==user_id) for user_id in user_id_set])
    num_pic_0p5_minus_ratio = np.array([np.abs(0.5-float(np.sum(Y[Y[:,1]==0,0]==user_id))/np.sum(Y[:,0]==user_id)) for user_id in user_id_set])
    
    print 'np.min(num_pic):',np.min(num_pic)
    print 'np.max(num_pic_0p5_minus_ratio):',np.max(num_pic_0p5_minus_ratio)

    user_ios = np.unique(Y[Y[:,1]==0,0])  
    user_and = np.unique(Y[Y[:,1]==1,0])
    user_intersect = np.intersect1d(user_ios,user_and)  
    user_ios_uni = np.setdiff1d(user_ios,user_intersect)
    user_and_uni = np.setdiff1d(user_and,user_intersect)

    print 'len(user_ios),len(user_and),len(user_intersect),len(user_ios_uni),len(user_and_uni)'
    print len(user_ios),len(user_and),len(user_intersect),len(user_ios_uni),len(user_and_uni)
    print 'np.median(user_intersect),np.mean(user_intersect<np.median(user_intersect))'
    print np.median(user_intersect),np.mean(user_intersect<np.median(user_intersect)) 

    flag_in_intersect = np.array([e in user_intersect for e in Y[:,0]])
    flag_less  = np.array([e < np.median(user_intersect) for e in Y[:,0]]) 
    flag_greater  = np.array([e >= np.median(user_intersect) for e in Y[:,0]])
 
    idx_test =  (flag_in_intersect & (Y[:,1]==0) & flag_less) |  (flag_in_intersect & (Y[:,1]==1) & flag_greater)
    idx_train = np.array([not e for e in idx_test])

    train_X, train_y_a, test_X, test_y_a = X[idx_train,:,:,:],Y[idx_train,:],X[idx_test,:,:,:],Y[idx_test,:]
    X = []
    Y = []
    train_X = train_X.astype(np.float32) / 255.
    test_X = test_X.astype(np.float32) / 255. 

    ATTR_NUM = train_y_a.shape[1]  
    CLASS_NUM = [len(np.unique(train_y_a[:,i])) for i in range(ATTR_NUM)]  
    #shuffle training data
    indices = np.arange(train_X.shape[0])
    np.random.shuffle(indices)
    train_X = train_X[indices]
    train_y_a = train_y_a[indices]
     
    input_shape = (64, 64, 3) 
 
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
    
    
    
    return train_X, train_y_a, test_X, test_y_a, train_y_c, test_y_c, ATTR_NUM, CLASS_NUM, input_shape,data_name,lambda_mat,prior_list

 
if __name__ == "__main__":
    train_X, train_y_a, test_X, test_y_a, train_y_c, test_y_c, ATTR_NUM, CLASS_NUM, input_shape,data_name,lambda_mat,prior_list = gen_raw_data(['Eyeglasses','Wearing_Hat'])
    user_id_set = [np.unique(train_y_a[:,i]) for i in range(train_y_a.shape[1])]
    print user_id_set
