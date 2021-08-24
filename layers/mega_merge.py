from keras.layers import Layer
import tensorflow as tf
from keras import backend as K
from keras.layers.advanced_activations import Softmax

class MegaMerge(Layer):
  def __init__(self, **kwargs):
    super(MegaMerge, self).__init__(**kwargs)

  def build(self, input_shape):
    super(MegaMerge, self).build(input_shape)

  def call(self,x):
    context,c2q,q2c=x
    self.G=K.concatenate([context,c2q,context*c2q,context*q2c],axis=-1)

    return self.G;

  def compute_output_shape(self, input_shape):
    return self.G.shape;