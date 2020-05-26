from keras.layers import Layer,LSTM,Bidirectional
import numpy as np
import tensorflow as tf

class ModellingLayer(Layer):
    def __init__(self,**kwargs):
        super(ModellingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.shape=input_shape
        print("dukche 3 build")

        self.lstm1 = Bidirectional(LSTM(int(input_shape[1]/8),
                                   activation='sigmoid',
                                   input_shape=(input_shape[0],input_shape[1]),
                                   return_sequences=True, trainable=True))
        self.lstm2 = Bidirectional(LSTM(int(input_shape[1]/8),
                                   activation='sigmoid',
                                   input_shape=(input_shape[0], int(input_shape[1]/4)),
                                   return_sequences=True, trainable=True))
        super(ModellingLayer, self).build(input_shape)

    def call(self,x):
        G=tf.reshape(x,shape=(1,self.shape[0],self.shape[1]))
        print("dukche 3")
        self.M1=self.lstm1(G)
        self.M2=self.lstm2(self.M1)

        G=tf.reshape(G,shape=(self.shape[0],self.shape[1]))
        self.M1=tf.reshape(self.M1,shape=(self.shape[0],int(self.shape[1]/4)))
        self.M2=tf.reshape(self.M2,shape=(self.shape[0],int(self.shape[1]/4)))

        temp1=tf.concat([G,self.M1],1)
        temp2=tf.concat([G,self.M2],1)

        temp=tf.concat([temp1,temp2],0)

        return tf.reshape(temp,shape=(2,self.shape[0],int(self.shape[1]*5/4)))

    def compute_output_shape(self, input_shape):
        print("dukche 3")
        return (2,input_shape[0],int((input_shape[1]*5)/4))