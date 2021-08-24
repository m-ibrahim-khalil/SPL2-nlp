from keras.layers import Layer
import tensorflow as tf
from keras import backend as K
from keras.layers.advanced_activations import Softmax

class C2Q_Layer(Layer):
  def __init__(self, **kwargs):
    super(C2Q_Layer, self).__init__(**kwargs)

  def build(self, input_shape):
    super(C2Q_Layer, self).build(input_shape)

  def call(self,x):
    similarity_matrix, question=x
    attention = tf.nn.softmax(similarity_matrix)

    self.U_A=K.sum(K.dot(attention,question),-2)

    return self.U_A

  def compute_output_shape(self, input_shape):
    return self.U_A.shape;