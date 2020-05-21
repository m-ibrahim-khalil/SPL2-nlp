from keras.layers import Layer
from keras.layers import LSTM,Bidirectional
import numpy as np
import tensorflow as tf

class C2VecLayer(Layer) :
    def __init__(self,**kwargs):
        super(C2VecLayer,self).__init__(**kwargs)

    def call(self,x):

        x = tf.cast(x, tf.float32)
        context,question=x[0],x[1]

        lstm=Bidirectional(LSTM(80,
                             activation='sigmoid',
                             input_shape=(766,80),
                             return_sequences=True))

        H=lstm(context)
        U=lstm(question)

        y=list()
        y.append(H)
        y.append(U)
        y=tf.convert_to_tensor(y)

        return y
