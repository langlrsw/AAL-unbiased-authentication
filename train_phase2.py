# -*- coding: utf-8 -*-

import numpy as np
import h5py
import os.path
from sklearn.metrics import roc_auc_score

from keras.models import Sequential, Model
from keras.layers import Convolution2D, MaxPooling2D, Dense, Dropout, Flatten, InputLayer, Input, merge,concatenate,add,Lambda,subtract,multiply
from keras import backend as K
from keras.utils.np_utils import to_categorical

from keras.models import Model
from keras.layers import Dense, Activation, Input, Reshape
from keras.layers import Conv1D, Flatten, Dropout
from keras.optimizers import SGD, Adam
from keras.layers.normalization import BatchNormalization

from model import reader_tensor
from model import reader_vector

from scipy.optimize import minimize_scalar

from util.util import label2uniqueID,split_test_as_valid,argmin_mean_FAR_FRR,auc_MTL,FAR_score,FRR_score,evaluate_result_valid,evaluate_result_test,mkdir
from util.util_keras import set_trainability
from nets.GD_basic import get_generative,get_discriminative
  
if K.backend() == "tensorflow":
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    config = K.tf.ConfigProto()
    config.gpu_options.allow_growth = True
#     config.gpu_options.per_process_gpu_memory_fraction = 0.3
    session = K.tf.Session(config=config)
    K.set_session(session)
    
 
  
def make_gan_phase_2_seen(GAN_in, G_set, D_set, loss,opt,loss_weights):
    output_set = []
    ATTR_NUM = len(G_set)
    for i in range(ATTR_NUM):
        G_out = []
        for j in range(ATTR_NUM):
            if i != j:
                set_trainability(G_set[j], False)
            else:
                set_trainability(G_set[j], True) 
            x = G_set[j](GAN_in[j])
            G_out.append(x)
        x = add(G_out) 
        set_trainability(D_set[i], True) 
        output_i = D_set[i](x)
        output_set.append(output_i)
    GAN = Model(GAN_in, output_set)
    GAN.compile(loss=loss, optimizer=opt, loss_weights=loss_weights)
    return GAN, output_set
 
def make_gan_phase_2_unseen(GAN_in, G_set, D_set, loss,opt,loss_weights):
    model_set = []
    output_set = []
    ATTR_NUM = len(G_set)
    for i in range(ATTR_NUM):
        G_out = []
        for j in range(ATTR_NUM):
            if i == j:
                set_trainability(G_set[j], False)
            else:
                set_trainability(G_set[j], True) 
            x = G_set[j](GAN_in[j])
            G_out.append(x)
        x = add(G_out) 
        set_trainability(D_set[i], True)
        output_i = D_set[i](x)
        output_set.append(output_i)
    GAN = Model(GAN_in, output_set)
    GAN.compile(loss=loss, optimizer=opt, loss_weights=loss_weights)
    return GAN, output_set


def mix_feature(feature, y_a,feature_dim,ATTR_NUM,CLASS_NUM):
    coef = [1]
    for i in range(ATTR_NUM):
        if i == ATTR_NUM - 1:
            break
        coef.append(coef[i]*CLASS_NUM[i])
        
    coef = np.array(coef)[:,np.newaxis]
    
    
    feats = [feature[:, i*feature_dim:(i+1)*feature_dim] for i in range(ATTR_NUM)]
    rand_idx = np.random.choice(range(len(y_a)), (200000, ATTR_NUM))
    mix_f = np.hstack(tuple(feats[i][rand_idx[:, i]] for i in range(ATTR_NUM)))
    mix_y = np.hstack(tuple(y_a[:, i][rand_idx[:, i], None] for i in range(ATTR_NUM))).dot(coef)
    mix_a = np.hstack(tuple(y_a[:, i][rand_idx[:, i], None] for i in range(ATTR_NUM))) 

    # mix_f = np.vstack((mix_f, feature))
    # mix_y = np.vstack((mix_y, y.dot([[100], [10], [1]])))
    return mix_f, mix_y, mix_a


def mix_feature_with_prob(feature,y_a,y_p,feature_dim,ATTR_NUM,CLASS_NUM,p_th,g_num = 200000):
    coef = [1]
    for i in range(ATTR_NUM):
        if i == ATTR_NUM - 1:
            break
        coef.append(coef[i]*CLASS_NUM[i])
        
    coef = np.array(coef)[:,np.newaxis]
    
    y_p_idxed = np.zeros(y_a.shape)
    for i in range(ATTR_NUM):
        y_p_idxed[:,i] = np.sum(to_categorical(y_a[:,i],CLASS_NUM[i])*y_p[i],axis=1)
        score = np.sort(y_p_idxed[:,i])
        p_th[i] = score[int(len(score)*0.05)]
        if i == 0:
            p_th[i] = 0.0
        print 'attribute %d, num above th = %d, ratio above th = %.4f' % (i,np.sum(y_p_idxed[:,i]>=p_th[i]),np.mean(y_p_idxed[:,i]>=p_th[i]))
    
    feats = [feature[:, i*feature_dim:(i+1)*feature_dim] for i in range(ATTR_NUM)]
    rand_idx = np.zeros((g_num,ATTR_NUM)).astype(int)
    for i in range(ATTR_NUM):
        rand_idx[:,i] = np.random.choice(np.where(y_p_idxed[:,i]>=p_th[i])[0], (g_num,))
    
    mix_f = np.hstack(tuple(feats[i][rand_idx[:, i]] for i in range(ATTR_NUM)))
    mix_y = np.hstack(tuple(y_a[:, i][rand_idx[:, i], None] for i in range(ATTR_NUM))).dot(coef)
    mix_a = np.hstack(tuple(y_a[:, i][rand_idx[:, i], None] for i in range(ATTR_NUM))) 
 
    return mix_f, mix_y, mix_a

def mix_feature2(feature, y_a, y_p, g_num, f_dim, a_num, a_c, m_pos, m_neg):

    a_pos_t = [np.where(y_a[:, idy] == 1)[0] for idy in range(a_num)]
    a_neg_t = [np.where(y_a[:, idy] == 0)[0] for idy in range(a_num)]
    a_pos_p = [np.where(y_p[idy][:, 1] > m_pos)[0] for idy in range(a_num)]
    a_neg_p = [np.where(y_p[idy][:, 0] > m_neg)[0] for idy in range(a_num)]

    a_pos = [list(set(p_t) & set(p_p)) for p_t, p_p in zip(a_pos_t, a_pos_p)]
    a_neg = [list(set(n_t) & set(n_p)) for n_t, n_p in zip(a_neg_t, a_neg_p)]

    mix_f = np.zeros((g_num * len(a_c), f_dim * a_num))
    mix_y = np.array([range(len(a_c)) for _ in range(g_num)]).T.flatten()

    for idx, acp in enumerate(a_c):
        tmp_f = np.zeros((g_num, f_dim * a_num))
        for idy, p in enumerate(acp):
            prob = np.random.rand(g_num)
            pos, neg = np.where(prob <= p)[0], np.where(prob > p)[0]
            idx_pos = np.random.choice(a_pos[idy] if len(a_pos[idy]) else a_pos_t[idy], len(pos))
            idx_neg = np.random.choice(a_neg[idy] if len(a_neg[idy]) else a_neg_t[idy], len(neg))
            tmp_f[pos, idy*f_dim:(idy+1)*f_dim] = feature[idx_pos, idy*f_dim:(idy+1)*f_dim]
            tmp_f[neg, idy*f_dim:(idy+1)*f_dim] = feature[idx_neg, idy*f_dim:(idy+1)*f_dim]
        mix_f[idx*g_num:(idx+1)*g_num] = tmp_f

    return mix_f, mix_y


def build_model(ATTR_NUM,CLASS_NUM,feature_dim,hidden_dim,hidden_dim_phase_2):
    G_set_phase_2 = []
    D_set_phase_2 = []   
    for i in range(ATTR_NUM):
        G,_ = get_generative(input_dim=feature_dim, out_dim=hidden_dim_phase_2,activity_l1 = 0.0)
        G_set_phase_2.append(G)
        D,_ = get_discriminative(input_dim=hidden_dim_phase_2, out_dim=CLASS_NUM[i],activation='softmax',kernel_l1=0.01)
        D_set_phase_2.append(D)
    
    #设置优化器
    opt = Adam(lr=1e-3)
    

    loss_weights = [1.]
    loss_weights.extend([0.1 for _ in range(ATTR_NUM-1)]) 

    GAN_in = [Input([feature_dim]) for _ in range(ATTR_NUM)] #设置第二阶段网络的输入，每个属性对应一个输入，维数为feature_dim
    
    GAN_phase_2_seen,_ = make_gan_phase_2_seen(GAN_in, G_set_phase_2, D_set_phase_2, 'categorical_crossentropy',opt,loss_weights)
#     GAN_phase_2_seen.summary() 

    GAN_phase_2_unseen,_ = make_gan_phase_2_unseen(GAN_in, G_set_phase_2, D_set_phase_2, 'categorical_crossentropy',opt,loss_weights)
#     GAN_phase_2_unseen.summary() 

    return GAN_phase_2_seen,GAN_phase_2_unseen
 
#第二阶段训练模型主函数    
def train(read_path):   
    
    
    hidden_dim_phase_2 = 128*3  
    batch_size = 32
    p_th = [0.0,0.98,0.98]
    g_num = 200000
     
    f = h5py.File(read_path, 'r')
    ATTR_NUM = int(np.array(f['ATTR_NUM']))
    CLASS_NUM = np.array(f['CLASS_NUM']).astype(int)
    
    train_f = np.array(f['train_f'])
    test_f = np.array(f['test_f'])  
    val_f = np.array(f['val_f']) 
    train_prob_set = [np.array(f['train_prob_%d' % i]) for i in range(ATTR_NUM)]
    test_prob_set = [np.array(f['test_prob_%d' % i]) for i in range(ATTR_NUM)]
    val_prob_set = [np.array(f['val_prob_%d' % i]) for i in range(ATTR_NUM)]
    train_y_a = np.array(f['train_y_a'])  
    test_y_a = np.array(f['test_y_a'])  
    val_y_a = np.array(f['val_y_a'])  
    train_y_c = np.array(f['train_y_c'])  
    test_y_c = np.array(f['test_y_c'])  
    val_y_c = np.array(f['val_y_c'])  
    hidden_dim = int(np.array(f['hidden_dim']))  
    feature_dim = int(np.array(f['feature_dim']))
    
    f.close()
    
#     print 'hidden_dim,feature_dim',hidden_dim,feature_dim
  
    GAN_phase_2_seen,GAN_phase_2_unseen = build_model(ATTR_NUM,CLASS_NUM,feature_dim,hidden_dim,hidden_dim_phase_2)   
     
    seen_set = np.unique(train_y_c)   
     
    for i_outer in range(10):
#         mix_f, mix_y, mix_a = mix_feature(train_f, train_y_a,feature_dim,ATTR_NUM,CLASS_NUM)
        mix_f, mix_y, mix_a = mix_feature_with_prob(train_f,train_y_a,train_prob_set,feature_dim,ATTR_NUM,CLASS_NUM,p_th,g_num)

        mix_f_seen, mix_y_seen, mix_a_seen = mix_f[np.array([e in seen_set for e in mix_y])], mix_y[np.array([e in seen_set for e in mix_y])], mix_a[np.array([e in seen_set for e in mix_y])] 

        mix_f_unseen, mix_y_unseen, mix_a_unseen = mix_f[np.array([e not in seen_set for e in mix_y])], mix_y[np.array([e not in seen_set for e in mix_y])], mix_a[np.array([e not in seen_set for e in mix_y])] 

        data_reader_seen = reader_vector.Reader(mix_f_seen,mix_a_seen, batch_size=batch_size)
        data_reader_unseen = reader_vector.Reader(mix_f_unseen,mix_a_unseen, batch_size=batch_size)
        
        n_seen,n_unseen = mix_f_seen.shape[0],mix_f_unseen.shape[0]
        n_ratio = n_unseen/n_seen
        n_step = n_seen/batch_size
        
        print n_step,n_ratio,n_seen,n_unseen
        print '*******************************************************************************'
        for i_step in range(n_step):
            x_batch, y_batch  = data_reader_seen.iterate_batch()
            GAN_phase_2_seen.train_on_batch([x_batch[:, ii*feature_dim:(ii+1)*feature_dim] for ii in range(ATTR_NUM)], [to_categorical(y_batch[:, i], CLASS_NUM[i]) for i in range(ATTR_NUM)])
            for _ in range(n_ratio):
                x_batch, y_batch  = data_reader_unseen.iterate_batch()
                GAN_phase_2_unseen.train_on_batch([x_batch[:, ii*feature_dim:(ii+1)*feature_dim] for ii in range(ATTR_NUM)], [to_categorical(y_batch[:, i], CLASS_NUM[i]) for i in range(ATTR_NUM)])
     
                
            #评估
            if (i_step == 10 or i_step == 20 or i_step == 50 or i_step == 100 or i_step == 200
               or i_step == 500 or i_step == 1000 or i_step == n_step-1):
                print '----------------- validation ------------------------'    
                prob_set = GAN_phase_2_seen.predict([val_f[:, ii*feature_dim:(ii+1)*feature_dim] for ii in range(ATTR_NUM)])
                dic_th_list = []
                for i_prob,prob in enumerate(prob_set):
                    label_test = to_categorical(val_y_a[:, i_prob], CLASS_NUM[i_prob])
                    dic_th,_ = evaluate_result_valid(label_test, prob,i_step,'outer step %d, attr %d' % (i_outer,i_prob) )
                    dic_th_list.append(dic_th)
                    break
                    
                print '----------------- test ------------------------'    
                prob_set = GAN_phase_2_seen.predict([test_f[:, ii*feature_dim:(ii+1)*feature_dim] for ii in range(ATTR_NUM)])
                for i_prob,prob in enumerate(prob_set):
                    label_test = to_categorical(test_y_a[:, i_prob], CLASS_NUM[i_prob])
                    evaluate_result_test(label_test, prob,i_step,'outer step %d, attr %d' % (i_outer,i_prob),dic_th_list[i_prob])
                    break
    return

if __name__ == "__main__":
  
    import os,signal,traceback
    try:
        save_root_path = 'transformed_feature_with_prob'
        read_path = '_colored_mnist_data_phase1_AUC_0.9573.h5'
        read_path = os.path.join(save_root_path,read_path)
        train(read_path)
    except:
        traceback.print_exc()
    finally:
        os.kill(os.getpid(),signal.SIGKILL)
 




