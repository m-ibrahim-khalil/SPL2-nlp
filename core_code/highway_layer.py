from keras import backend as K
import tensorflow as tf
import keras
from keras.engine.topology import Layer
from keras.layers import Dense, Activation, Multiply, Add, Lambda, TimeDistributed
from keras.initializers import Constant
import core_code.word_embedding as we
import core_code.word_char_embd as wce
import numpy as np


class Highway(Layer):

    activation = None
    transform_gate_bias = None

    def __init__(self, activation='relu', transform_gate_bias=-1, **kwargs):
        self.activation = activation
        self.transform_gate_bias = transform_gate_bias
        self.dense_1 = None
        self.dense_2 = None
        self.supports_masking = True
        super(Highway, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        dim = input_shape[-1]
        transform_gate_bias_initializer = Constant(self.transform_gate_bias)
        self.dense_1 = Dense(units=dim, bias_initializer=transform_gate_bias_initializer)
        self.dense_1.build(input_shape)
        self.dense_2 = Dense(units=dim)
        self.dense_2.build(input_shape)
        self.trainable_weights = self.dense_1.trainable_weights + self.dense_2.trainable_weights

        super(Highway, self).build(input_shape)  # Be sure to call this at the end

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask = None):
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            x *= K.expand_dims(mask, axis=-1)
        dim = K.int_shape(x)[-1]
        transform_gate = self.dense_1(x)
        transform_gate = Activation("sigmoid")(transform_gate)
        carry_gate = Lambda(lambda x: 1.0 - x, output_shape=(dim,))(transform_gate)
        transformed_data = self.dense_2(x)
        transformed_data = Activation(self.activation)(transformed_data)
        transformed_gated = Multiply()([transform_gate, transformed_data])
        identity_gated = Multiply()([carry_gate, x])
        value = Add()([transformed_gated, identity_gated])
        return value

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config['activation'] = self.activation
        config['transform_gate_bias'] = self.transform_gate_bias
        return config


if __name__ == '__main__':
    highway_layer = Highway(transform_gate_bias=-2)
    directory = "F://Pycharm Projects//glove.6B.50d.txt"
    output = we.get_glove(directory, 50)
    emb_matrix = output[0]
    word2id = output[1]
    id2word = output[2]
    word_dict = word2id
    char_dict, _, _ = wce.create_char_dicts()
    # question_layer = TimeDistributed(highway_layer, name=highway_layer.name + "_qtd")
    # question_embedding = question_layer(emb_matrix[word2id['what']])
    # print(question_embedding)
    emb = np.array([[emb_matrix[word2id['what']],emb_matrix[word2id['what']]]])
    print(emb)
    #print(tf.shape(emb))
    emb = tf.convert_to_tensor(emb, dtype=tf.float32)
    print(tf.shape(emb))
    print(emb)
    print(emb.shape)
    highway_layer = keras.layers.Dense(units=5, name='d1')(highway_layer)
    question_embedding = highway_layer(emb)
    print(question_embedding)
    print(highway_layer.losses)