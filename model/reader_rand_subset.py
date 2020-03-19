import threading
from time import sleep
import numpy
import numpy as np

test_fold = 1


# from utils import *


class Reader(threading.Thread):
    """ This class is designed to automatically feed mini-batches.
        The reader constantly monitors the state of the variable 'data_buffer'.
        When finding the 'data_buffer' is None, the reader will fill a mini-batch into it.
        This is done in the backend, i.e., the reader is in an independent thread.
        For users, they only need to call iterate_batch() to get a new mini-batch.
    """

    # Initialize the super class
    def __init__(self, x_train, y_train,ratio_subset = 0.92, rng_seed=123, n_nearby=128, batch_size=32, center_pad=True, valid_fold=None):
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
        self.n_nearby = n_nearby
        self.batch_size = batch_size
        self.center_pad = center_pad

        # train_data : {'data', 'label', 'file', 'fold', 'salience', 'sound'}
        # train_data['data'] is an ndarray, T x D,
        # each row a sample for a certain time-stamp

        self.x_train = x_train
        self.y_train = y_train
        self.dim_time = len(x_train[0])
        self.dim_feature = len(x_train[0][0])
        self.dim_class_num = len(y_train[0])
        self.ratio_subset = ratio_subset

        # Shuffle the data
        # We just need to shuffle 'query_index'
        # at each beginning of a new epoch
        # 'shuffle_index' is a list indicating
        # all the positions in 'query_index'
        self.shuffle_index = range(len(self.x_train))

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
                    self.shuffle_index = range(len(self.x_train))
                    self.rng.shuffle(self.shuffle_index)
                    self.index_start = 0
                else:
                    # This case means index_start == len(shuffle_index)
                    # Thus, we've finished this epoch
                    # let's shuffle it again.
                    self.shuffle_index = range(len(self.x_train))
                    self.rng.shuffle(self.shuffle_index)
                    batch_index = self.shuffle_index[0: self.batch_size]
                    self.index_start = self.batch_size

                final_dim_time = int(self.dim_time*self.ratio_subset)
                data = np.zeros((len(batch_index), final_dim_time, self.dim_feature))
                label = np.zeros((len(batch_index), self.dim_class_num))

                
                
                if self.ratio_subset < 1:
                    for i in range(len(batch_index)):
                        end_point = np.random.randint(0,int(self.dim_time*(1-self.ratio_subset))) + 1
#                         print self.x_train[batch_index[i]].shape
#                         print end_point
#                         print final_dim_time

                        data[i] = self.x_train[batch_index[i]][-final_dim_time-end_point:-end_point,:]

                        label[i] = self.y_train[batch_index[i]]
                else:
                    for i in range(len(batch_index)):
                        start_point = 0

                        data[i] = self.x_train[batch_index[i]][start_point:start_point+final_dim_time,:]

                        label[i] = self.y_train[batch_index[i]]
   

                with self.lock:
                    self.data_buffer = data, label
            sleep(0.0001)

    def iterate_batch(self):
        while self.data_buffer is None:
            sleep(0.0001)

        data, label = self.data_buffer
        data = numpy.asarray(data, dtype=numpy.float32)
        label = numpy.asarray(label, dtype=numpy.int32)
        with self.lock:
            self.data_buffer = None

        return data, label

    def close(self):
        self.running = False
        self.join()


if __name__ == '__main__':
    """ Let's just write the test function
        for reader in the same file
    """
    test_fold = 1
    valid_folds = [2]
    rng_seed = 123
    x_train = numpy.zeros((120, 1500, 132))
    y_train = numpy.zeros((120, 10))


    model = Reader(x_train, y_train)

    for i in range(10):
        print('Loading %d-th mini-batch ...' % i)
        data, label = model.iterate_batch()

        print('max: %f, min: %f' % (data.max(), data.min()))
        print (label.shape)
        print (data.shape)

    model.close()



