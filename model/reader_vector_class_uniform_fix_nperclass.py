import threading
from time import sleep
import numpy
import numpy as np

test_fold = 1

import time as time_simple
    
import Queue
from collections import defaultdict


    
    
    


class Reader(threading.Thread):
    """ This class is designed to automatically feed mini-batches.
        The reader constantly monitors the state of the variable 'data_buffer'.
        When finding the 'data_buffer' is None, the reader will fill a mini-batch into it.
        This is done in the backend, i.e., the reader is in an independent thread.
        For users, they only need to call iterate_batch() to get a new mini-batch.
    """

    # Initialize the super class
    def __init__(self, x_train, y_train,ratio_subset = 1., rng_seed=123, batch_size=32,n_sample_per_class=2, flag_shuffle=True):
        """

        Parameters:
        ----------
        :param test_fold   : int, for testing
        :param valid_folds : list of ints, validation set
        :param rng_seed    : random seed to make our code more controllable
        :param n_nearby    : number of time stamps to consider
        :param batch_size  : mini-batch
        :param center_pad  : center pad the mls feature with 0's
                             Actually, we should pad with -80.
        :param valid_fold  : int, for validation
        """

        threading.Thread.__init__(self)

        # Settings
        self.rng = numpy.random.RandomState(seed=rng_seed)
        self.batch_size = batch_size
   

        # train_data : {'data', 'label', 'file', 'fold', 'salience', 'sound'}
        # train_data['data'] is an ndarray, T x D,
        # each row a sample for a certain time-stamp

        self.x_train = x_train
        self.y_train = y_train
        self.dim_time = len(x_train[0])
        self.dim_class_num = len(y_train[0])
        self.ratio_subset = ratio_subset
        self.flag_shuffle = flag_shuffle
        self.n_sample = y_train.shape[0]
        self.n_sample_per_class = n_sample_per_class
        

        # Shuffle the data
        # We just need to shuffle 'query_index'
        # at each beginning of a new epoch
        # 'shuffle_index' is a list indicating
        # all the positions in 'query_index'
        
#         start_time = time_simple.time()
        self.shuffle_index = self.gen_idx()#range(len(self.x_train))
#         end_time = time_simple.time()
#         print 'process time: %.3f' % (end_time - start_time)

        if self.flag_shuffle:
            self.rng.shuffle(self.shuffle_index)

        self.index_start = 0

        # Initialization
        self.running = True
        self.data_buffer = None
        self.lock = threading.Lock()

        # Start thread
        self.start()

    def run(self):
        """ Overwrite the 'run' method of threading.Thread
        """
        while self.running:
            if self.data_buffer is None:
                if self.index_start + self.batch_size <= len(self.shuffle_index):
                    # This case means we are still in this epoch
                    batch_index = self.shuffle_index[self.index_start: self.index_start + self.batch_size]
                    self.index_start += self.batch_size

                elif self.index_start < len(self.shuffle_index):
                    # This case means we've come to the
                    # end of this epoch, take all the rest data
                    # and shuffle the training data again
                    batch_index = self.shuffle_index[self.index_start:]

                    # Now, we've finished this epoch
                    # let's shuffle it again.
                    self.shuffle_index = self.gen_idx()#range(len(self.x_train))
                    if self.flag_shuffle:
                        self.rng.shuffle(self.shuffle_index)
                    self.index_start = 0
                    
                else:
                    # This case means index_start == len(shuffle_index)
                    # Thus, we've finished this epoch
                    # let's shuffle it again.
                    self.shuffle_index = self.gen_idx()#range(len(self.x_train))
                    if self.flag_shuffle:
                        self.rng.shuffle(self.shuffle_index)
                    batch_index = self.shuffle_index[0: self.batch_size]
                    self.index_start = self.batch_size
 
                data = self.x_train[:len(batch_index)].copy()
                label = self.y_train[:len(batch_index)].copy()

                
                
                
                for i in range(len(batch_index)):
               

                    data[i] = self.x_train[batch_index[i]] 

                    label[i] = self.y_train[batch_index[i]]
   

                with self.lock:
                    self.data_buffer = data, label
            sleep(0.0001)

    def iterate_batch(self):
        while self.data_buffer is None:
            sleep(0.0001)

        data, label = self.data_buffer
        data = numpy.asarray(data, dtype=numpy.float32)
#         label = numpy.asarray(label, dtype=numpy.int32)
        with self.lock:
            self.data_buffer = None

        return data, label

    def close(self):
        self.running = False
        self.join()
        
    def gen_idx(self):
        
        y_train = self.y_train[:,0]
        n = len(y_train)
 
        idx = np.arange(n)
        class_set,class_num = np.unique(y_train,return_counts=True)
        dic_user2sample_num = {class_set[i]: class_num[i] for i in range(len(class_set))}
        dic_user2indices = defaultdict(list)

        n_sample_per_class = self.n_sample_per_class
  
        idx_ret = []
        self.rng.shuffle(idx)
        for i in range(n):
            item = idx[i]
            cur_id = y_train[item]
            
            if dic_user2sample_num[cur_id] == 1:
                idx_ret.append(item)
            elif dic_user2sample_num[cur_id]<n_sample_per_class:
                cur_list = dic_user2indices[cur_id]
                if len(cur_list) == dic_user2sample_num[cur_id]-1:
                    idx_ret.extend(cur_list)
                    idx_ret.append(item)
                    dic_user2indices[cur_id] = []
                else:
                    cur_list.append(item)
            else:
                cur_list = dic_user2indices[cur_id]
                if len(cur_list) == n_sample_per_class-1:
                    idx_ret.extend(cur_list)
                    idx_ret.append(item)
                    dic_user2indices[cur_id] = []
                    dic_user2sample_num[cur_id] -= n_sample_per_class
                else:
                    cur_list.append(item)
 
        return idx_ret
            
 


