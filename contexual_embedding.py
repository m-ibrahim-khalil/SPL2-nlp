#dynamic

from keras.layers import Layer
from keras.layers import LSTM,Bidirectional
import tensorflow as tf

class C2VecLayer(Layer) :
    def __init__(self,**kwargs):
        super(C2VecLayer,self).__init__(**kwargs)

    def build(self, input_shape):
        self.shape=input_shape
        print("dukche 1")

        self.lstm = Bidirectional(LSTM(input_shape[3],
                                  activation='sigmoid',
                                  input_shape=(input_shape[2], input_shape[3]),
                                  return_sequences=True, trainable=True))
        super(C2VecLayer, self).build(input_shape)

    def call(self,x):

        x = tf.cast(x, tf.float32)
        context=x[0]
        question=x[1]
        print("dukche 1")
        H=self.lstm(context)
        U=self.lstm(question)

        y=tf.concat([H,U],0)

        return tf.reshape(y,shape=(2,1,self.shape[2],self.shape[3]*2))

    def compute_output_shape(self, input_shape):
        print("dukche 1")
        return (input_shape[0],input_shape[1],input_shape[2],input_shape[3]*2)
