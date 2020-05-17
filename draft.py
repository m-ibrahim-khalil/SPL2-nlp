import tensorflow as tf
from keras.layers import Layer

class MyLayer1(Layer):
    def __init__(self,**kwargs):
        super(MyLayer1,self).__init__(**kwargs)

    def call(self, x):
        return x

class MyLayer2(Layer):
    def __init__(self,**kwargs):
        super(MyLayer2,self).__init__(**kwargs)

    def call(self, x):
        temp_list=list()
        a=tf.ones([1,])
        b=tf.tensordot(x,a,1)
        temp_list.append(b.numpy()) # I stucked here. I can't extract the value of b.
        return x

x=tf.Variable(tf.zeros([1,]))
myLayer1=MyLayer1()
myLayer2=MyLayer2()

y=myLayer1(x)
myLayer2(y)




'''from Word_embedding import W2VecLayer
from contexual_embedding import C2VecLayer
from BiDAF import BiAttentionLayer

w2vec=W2VecLayer(5)
x=w2vec.call(['i am dip','who i'])
c2vec=C2VecLayer()
y=c2vec(x)
bidaf = BiAttentionLayer()
z=bidaf(y)'''

