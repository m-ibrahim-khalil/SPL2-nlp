from keras.layers import Layer
import tensorflow as tf
from keras import backend as K


class OutputLayer(Layer):
    def __init__(self, **kwargs):
        super(OutputLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.w1 = self.add_weight(name="w1",
                                  shape=(input_shape[0][2],),
                                  initializer='uniform',
                                  trainable=True)
        self.w2 = self.add_weight(name="w2",
                                  shape=(input_shape[0][2],),
                                  initializer='uniform',
                                  trainable=True)

        super(OutputLayer, self).build(input_shape)

    def call(self, x):
        answer_span1 = tf.tensordot(x[0], tf.transpose(self.w1), 1)
        answer_span2 = tf.tensordot(x[1], tf.transpose(self.w2), 1)

        self.p1 = tf.nn.softmax(answer_span1)
        self.p2 = tf.nn.softmax(answer_span2)

        return self.p1, self.p2

    def compute_output_shape(self, input_shape):
        return self.p1.shape, self.p2.shape
