import numpy as np

def my_dense(x,W,b):
    return np.dot(x,W)+b[np.newaxis,:]
def my_sigmoid(x,alpha=1):
    return 1./(1.+np.exp(-alpha*x))
def my_swish(x):
    return x*my_sigmoid(x)
def my_undropout(x,p=0.5):
    return x*p
def my_batchnorm(x,gamma,beta,moving_mean,moving_variance):
    inv = 1./np.sqrt(moving_variance + 1e-3)
    inv = inv * gamma
    return x * inv + beta - moving_mean * inv
def my_tanh(x):
    return my_sigmoid(x,alpha=2) * 2. - 1.

