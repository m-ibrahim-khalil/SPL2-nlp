from keras.layers import Layer
import tensorflow as tf
from keras import backend as K
from keras.layers.advanced_activations import Softmax


class Q2C_Layer(Layer):
    def __init__(self, **kwargs):
        super(Q2C_Layer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Q2C_Layer, self).build(input_shape)

    def call(self, x):
        similarity_matrix, context = x
        attention = tf.nn.softmax(K.max(similarity_matrix, axis=-1))

        temp = K.expand_dims(K.sum(K.dot(attention, context), -2), 1)

        H_A = K.tile(temp, [1, similarity_matrix.shape[1], 1])

        return H_A

    def compute_output_shape(self, input_shape):
        return self.H_A.shape