from keras.layers import Layer, LSTM, Bidirectional
import numpy as np
import tensorflow as tf


class ModellingLayer(Layer):
    def __init__(self, **kwargs):
        super(ModellingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.shape = input_shape

        self.lstm1 = Bidirectional(LSTM(int(input_shape[2] // 8),
                                        activation='tanh',
                                        input_shape=(input_shape[1], input_shape[2]),
                                        return_sequences=True, trainable=True))
        self.lstm2 = Bidirectional(LSTM(int(input_shape[2] // 8),
                                        activation='tanh',
                                        input_shape=(input_shape[1], int(input_shape[2] // 4)),
                                        return_sequences=True, trainable=True))
        super(ModellingLayer, self).build(input_shape)

    def call(self, x):
        self.M1 = self.lstm1(x)

        self.M2 = self.lstm2(self.M1)

        self.temp1 = tf.concat([x, self.M1], -1)
        self.temp2 = tf.concat([x, self.M2], -1)

        return self.temp1, self.temp2

    def compute_output_shape(self, input_shape):
        return self.temp1.shape, self.temp2.shape