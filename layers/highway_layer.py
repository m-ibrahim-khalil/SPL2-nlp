import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Activation, Multiply, Add, Lambda
from tensorflow.keras.layers import Layer
from keras.initializers import Constant
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
K.set_floatx('float64')


class Highway(Layer):
    def __init__(self, activation='tanh', transform_gate_bias=-3, **kwargs):
        self.activation = activation
        self.transform_gate_bias = transform_gate_bias
        super(Highway, self).__init__(**kwargs)

    def build(self, input_shape):
        dim = input_shape[-1]
        # print("Highway Layer input_shape and dim: " + str(input_shape), dim)
        transform_gate_bias_initializer = Constant(self.transform_gate_bias)
        self.dense_1 = Dense(units=dim, bias_initializer=transform_gate_bias_initializer)
        self.dense_2 = Dense(units=dim)
        super(Highway, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        # print("Highway layer input: "+ str(x))
        dim = K.int_shape(x)[-1]
        # print("Highway dim: " + str(dim))
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
        # print("output of HighwayLayer: "+str(input_shape))
        return input_shape

    def get_config(self):
        config = super().get_config()
        config['activation'] = self.activation
        config['transform_gate_bias'] = self.transform_gate_bias
        return config