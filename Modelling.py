from keras.layers import Layer,LSTM,Bidirectional
import numpy as np
import tensorflow as tf

class ModellingLayer(Layer):
    def __init__(self,**kwargs):
        super(ModellingLayer, self).__init__(**kwargs)

    def call(self,x):
        G=tf.reshape(x,shape=(1,5,400))

        lstm1=Bidirectional(LSTM(50,
                                   activation='sigmoid',
                                   input_shape=(5,400),
                                   return_sequences=True))
        lstm2=Bidirectional(LSTM(50,
                                   activation='sigmoid',
                                   input_shape=(5,100),
                                   return_sequences=True))

        self.M1=lstm1(G)
        self.M2=lstm2(self.M1)

        G=tf.reshape(G,shape=(5,400))
        self.M1=tf.reshape(self.M1,shape=(5,100))
        self.M2=tf.reshape(self.M2,shape=(5,100))

        temp1=tf.concat([G,self.M1],1)
        temp2=tf.concat([G,self.M2],1)

        temp=tf.concat([temp1,temp2],0)

        return tf.reshape(temp,shape=(2,5,500))