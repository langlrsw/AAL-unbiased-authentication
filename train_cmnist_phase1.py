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

from model import reader_tensor
from model import reader_vector

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
from model import resnet2
# from data.gendata_classic_grouped_classes import gen_raw_data
# from data.gendata_PIE_grouped_classes import gen_raw_data
# from data.gendata_celebA_grouped_classes import gen_raw_data
# from data.gendata_colormnist_grouped_classes_general import gen_raw_data
from data.gendata_colormnist_grouped_classes import gen_raw_data


from nets.nets import input_model_simpleCNN as input_model
 


from nets.GD_basic import get_generative,get_discriminative

from util.util import label2uniqueID,split_test_as_valid,argmin_mean_FAR_FRR,auc_MTL,FAR_score,FRR_score,evaluate_result_valid,evaluate_result_test,mkdir
from util.util_keras import set_trainability
 
from util.util import label2uniqueID,split_test_as_valid,argmin_mean_FAR_FRR,auc_MTL,FAR_score,FRR_score,evaluate_result_valid,evaluate_result_test,mkdir,split_train_test
from util.util_keras import set_trainability,my_get_shape,outer_product,prob2extreme
from util.util import my_zscore_test,my_zscore
from sklearn.semi_supervised import LabelPropagation
from losses.losses import *
 
    
if K.backend() == "tensorflow":
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    config = K.tf.ConfigProto()
    config.gpu_options.allow_growth = True
#     config.gpu_options.per_process_gpu_memory_fraction = 0.95
    session = K.tf.Session(config=config)
    K.set_session(session)
    
def make_gan(inputs, G, D, G_trainable,D_trainable):
    set_trainability(G, G_trainable)
    set_trainability(D, D_trainable)
    x = G(inputs)
    output = D(x)
    return output

def make_gan_phase_1_gen_hidden_feature(inputs, G_in, G_set):
    output_set = []
    ATTR_NUM = len(G_set)
    for i in range(ATTR_NUM):
        x = G_set[i](G_in)
        output_set.append(x)
    feats = concatenate(output_set)
    GAN = Model(inputs, feats)
    return GAN, feats

def make_gan_phase_1_task(inputs, GAN_in, G_set, D_set, loss,opt,loss_weights):
    output_set = []
    ATTR_NUM = len(G_set)

    G_trainable = True
    D_trainable = True
    for i in range(ATTR_NUM):  
        output_ii = make_gan(GAN_in, G_set[i], D_set[i][i], G_trainable,D_trainable)
        output_set.append(output_ii)
    model = Model(inputs, output_set)
    model.compile(loss=loss, optimizer=opt, loss_weights=loss_weights)
    return model, output_set

def make_gan_phase_1_domain_pos(inputs, GAN_in, G_set, D_set, loss,opt,loss_weights):
    output_set = []
    ATTR_NUM = len(G_set)
    G_trainable = False 
    D_trainable = True
    for i in range(ATTR_NUM):
        for j in range(ATTR_NUM):
            if i == j:
                D_trainable = False  
            else:
                D_trainable = True  
            output_ij = make_gan(GAN_in, G_set[i], D_set[i][j], G_trainable,D_trainable)
            output_set.append(output_ij)
    GAN = Model(inputs, output_set)
    GAN.compile(loss=loss, optimizer=opt, loss_weights=loss_weights)
    return GAN, output_set

def make_gan_phase_1_domain_neg(inputs, GAN_in, G_set, D_set, loss,opt,loss_weights):
    output_set = []
    ATTR_NUM = len(G_set)
    D_trainable = False  
    G_trainable = True
    for i in range(ATTR_NUM):
        for j in range(ATTR_NUM):
            if i == j:
                G_trainable = False  
            else:
                G_trainable = True  
            output_ij = make_gan(GAN_in, G_set[i], D_set[i][j], G_trainable,D_trainable)
            output_set.append(output_ij)
    GAN = Model(inputs, output_set)
    GAN.compile(loss=loss, optimizer=opt, loss_weights=loss_weights)
    return GAN, output_set
 
def build_model(ATTR_NUM,CLASS_NUM,feature_dim,hidden_dim,input_shape,lambda_mat):

    model_input, inputs, _, shared_dim = input_model(hidden_dim,input_shape)
       
    G_set_phase_1 = []
    D_set_phase_1 = []   
    for i in range(ATTR_NUM):
        G,_ = get_generative(input_dim=shared_dim, out_dim=feature_dim)
        G_set_phase_1.append(G)
        D_set_sub = []
        for j in range(ATTR_NUM):
            if i == j:
                activation = 'softmax'  
            else:
                activation = 'softmax'  
            D,_ = get_discriminative(input_dim=feature_dim, out_dim=CLASS_NUM[j],activation=activation)
            D_set_sub.append(D)
        D_set_phase_1.append(D_set_sub)
 
    opt_gan = Adam(lr=0.0002,beta_1=0.5,beta_2=0.999)
    opt = Adam(lr=1e-3)
    
    loss_weights = [1.]
    loss_weights.extend([0.1 for _ in range(ATTR_NUM-1)]) 
    
    set_trainability(model_input, True) 
    
    feats = model_input(inputs)
    loss = [K.categorical_crossentropy for _ in range(ATTR_NUM)]
    GAN_phase_1_task,_ = make_gan_phase_1_task(inputs, feats, G_set_phase_1, D_set_phase_1, loss,opt,loss_weights)

 
    for i in range(ATTR_NUM):
        loss_weights = [1.]
        loss_weights.extend([0.1 for _ in range(ATTR_NUM-1)]) 
        for j in range(ATTR_NUM):
            if i!=j:
                loss_weights[j] = loss_weights[j] * lambda_mat[j,i]
        if i == 0:
            loss_w = loss_weights
        else:
            loss_w.extend(loss_weights)
    loss_weights = loss_w

    set_trainability(model_input, False)
    feats = model_input(inputs)
    GAN_phase_1_domain_pos,_ = make_gan_phase_1_domain_pos(inputs, feats, G_set_phase_1, D_set_phase_1, 'mse',opt_gan,loss_weights)
    set_trainability(model_input, True)
    feats = model_input(inputs)
    GAN_phase_1_domain_neg,_ = make_gan_phase_1_domain_neg(inputs, feats, G_set_phase_1, D_set_phase_1, 'mse',opt_gan,loss_weights)
 
    Model_gen_hidden_feature,_=make_gan_phase_1_gen_hidden_feature(inputs, feats, G_set_phase_1)

    return GAN_phase_1_task,GAN_phase_1_domain_pos,GAN_phase_1_domain_neg,Model_gen_hidden_feature


#第一阶段训练模型主函数            
def train():   
 
    color_id_1,color_id_2 = 6,3
 
    hidden_dim = 128*10 #useless for cmnist data
    feature_dim = 128 
    batch_size = 32
    n_step = 100000000
    save_name = 'dense' 

    train_X, train_y_a, test_X, test_y_a, train_y_c, test_y_c, ATTR_NUM, CLASS_NUM, input_shape,data_name,lambda_mat,prior_list = gen_raw_data()
    
    idx_valid,idx_test = split_test_as_valid(test_y_c)
    val_X = test_X[idx_valid]
    val_y_a = test_y_a[idx_valid]
    val_y_c = test_y_c[idx_valid]

    test_X = test_X[idx_test]
    test_y_a = test_y_a[idx_test]
    test_y_c = test_y_c[idx_test]
 
    print 'start building model'
    GAN_phase_1_task,GAN_phase_1_domain_pos,GAN_phase_1_domain_neg,Model_gen_hidden_feature = build_model(ATTR_NUM,CLASS_NUM,feature_dim,hidden_dim,input_shape,lambda_mat)   
    
    if np.size(input_shape) == 1:
        data_reader_train = reader_vector.Reader(train_X,train_y_a, batch_size=batch_size)  
    elif np.size(input_shape) == 3:
        data_reader_train = reader_tensor.Reader(train_X,train_y_a, batch_size=batch_size)  
    
    print 'start training'
    for i_step in range(n_step):
        x_batch, y_batch  = data_reader_train.iterate_batch()
         
 
        GAN_phase_1_task.train_on_batch(x_batch,[to_categorical(y_batch[:, i], CLASS_NUM[i]) for i in range(ATTR_NUM)])
        
  
        y_batch_domain = []
        y_batch_domain_inv = []
        for i in range(ATTR_NUM):
            for j in range(ATTR_NUM):
                yy = to_categorical(y_batch[:, j], CLASS_NUM[j])
                yyy = yy.copy()
                y_batch_domain.append(yyy)
                if i == j:
                    y_batch_domain_inv.append(yyy)  
                else:
#                     y_batch_domain_inv.append(1-yyy) 
 
                    prior_vec = prior_list[j].copy()
                    prior_vec = np.ones((yyy.shape[0],1))*prior_vec[np.newaxis,:]
                    prior_vec[yyy==1] = 0
                    zzz = prior_vec/np.sum(prior_vec,axis=1)[:,np.newaxis]
                    zzz = prior_vec * CLASS_NUM[j]
                    
                    y_batch_domain_inv.append(zzz) 
        
        
 
        GAN_phase_1_domain_pos.train_on_batch(x_batch,y_batch_domain)
        
 
        for _ in range(5):
            GAN_phase_1_domain_neg.train_on_batch(x_batch,y_batch_domain_inv)
        
        #评估
        if (i_step == 100 or i_step == 200 or i_step == 500 or i_step == 1000 or i_step == 2000 or i_step == 5000 or i_step == 10000
            or i_step == 20000 or i_step == 50000 or i_step == 100000 or i_step == n_step -1):
            
 
            
            prob_set = GAN_phase_1_task.predict(val_X)
            val_prob_set = prob_set
            print '---------------- validation --------------------------'
            auc_record = []
            dic_th_list = []
            for i_prob,prob in enumerate(prob_set):
                label_test = to_categorical(val_y_a[:, i_prob], CLASS_NUM[i_prob])             
                dic_th,auc_main_task = evaluate_result_valid(label_test, prob,i_step,'attr %d' % i_prob)
                dic_th_list.append(dic_th)
                auc_record.append(auc_main_task)
                break
             
            prob_set = GAN_phase_1_task.predict(test_X)
            test_prob_set = prob_set
            print '---------------- test ----------------------'
            for i_prob,prob in enumerate(prob_set):
                label_test = to_categorical(test_y_a[:, i_prob], CLASS_NUM[i_prob])             
                evaluate_result_test(label_test, prob,i_step,'attr %d' % i_prob,dic_th_list[i_prob])
                break
  
       
            train_f = Model_gen_hidden_feature.predict(train_X)
            val_f = Model_gen_hidden_feature.predict(val_X)
            test_f = Model_gen_hidden_feature.predict(test_X)
            
            train_prob_set = GAN_phase_1_task.predict(train_X)
            
 
            save_root_path = 'transformed_feature_with_prob'
            mkdir(save_root_path)
            write_path = '%s_%s_data_phase1_AUC_%.4f.h5' % (save_name,data_name,auc_record[0])
            print write_path
            f = h5py.File(os.path.join(save_root_path,write_path), 'w')
            f['train_f'] = train_f
            f['val_f'] = val_f
            f['test_f'] = test_f
            for i in range(ATTR_NUM):
                f['train_prob_%d' % i] = train_prob_set[i]
                f['val_prob_%d' % i] = val_prob_set[i]
                f['test_prob_%d'% i] = test_prob_set[i]
            f['train_y_a'] = train_y_a
            f['test_y_a'] = test_y_a
            f['val_y_a'] = val_y_a
            f['train_y_c'] = train_y_c
            f['test_y_c'] = test_y_c
            f['val_y_c'] = val_y_c
            f['hidden_dim'] = hidden_dim
            f['feature_dim'] = feature_dim
            f['ATTR_NUM'] = ATTR_NUM
            f['CLASS_NUM'] = np.array(CLASS_NUM)
           
            f.close()
        
     
    return
         
     
if __name__ == "__main__":

 
    import os,signal,traceback
    try:
        train()
    except:
        traceback.print_exc()
    finally:
        os.kill(os.getpid(),signal.SIGKILL)
    
    










