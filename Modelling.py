from keras.layers import Layer,LSTM,Bidirectional
import numpy as np
import tensorflow as tf

class ModellingLayer(Layer):
    def __init__(self,**kwargs):
        super(ModellingLayer, self).__init__(**kwargs)

    def call(self,x):
        shape = tf.shape(x)
        G=tf.reshape(x,shape=(1,shape[0],shape[1]))

        lstm1=Bidirectional(LSTM(shape[1]/8,
                                   activation='sigmoid',
                                   input_shape=(shape[0],shape[1]),
                                   return_sequences=True))
        lstm2=Bidirectional(LSTM(shape[1]/8,
                                   activation='sigmoid',
                                   input_shape=(shape[0],shape[1]/4),
                                   return_sequences=True))

        self.M1=lstm1(G)
        self.M2=lstm2(self.M1)

        y=list()
        y.append(np.concatenate((G,self.M1)))
        y.append(np.concatenate((G,self.M2)))

        return tf.convert_to_tensor(y)