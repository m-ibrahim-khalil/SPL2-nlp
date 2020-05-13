from keras.layers import Layer,LSTM,Bidirectional
import numpy as np

class ModellingLayer(Layer):
    def __init__(self,**kwargs):
        super(ModellingLayer, self).__init__(**kwargs)

    def call(self,x):
        G=np.array(x)
        G=G.reshape((1,G.shape[0],G.shape[1]))

        self.M1=Bidirectional(LSTM(G.shape[2]/8,
                                   activation='sigmoid',
                                   input_shape=(G.shape[1],G.shape[2]),
                                   return_sequences=True))
        self.M2=Bidirectional(LSTM(self.M1.input_shape[2]/2,
                                   activation='sigmoid',
                                   input_shape=(self.M1.shape[1],self.M1.shape[2]),
                                   return_sequences=True))

        y=list()
        y.append(self.M1)
        y.append(self.M2)

        return y