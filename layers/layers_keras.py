from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

class NegDistLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        distmat = K.repeat_elements(K.sum(K.pow(x, 2),axis=1, keepdims=True),self.output_dim,axis=1) + \
                   K.repeat_elements(K.expand_dims(K.sum(K.pow(self.kernel, 2),axis=0),axis=0),input_shape[0],axis=0)
        distmat = distmat - 2. * K.dot(x, self.kernel)
        return -K.sqrt(K.clip(distmat,1e-12,None))
        
     
#         return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)