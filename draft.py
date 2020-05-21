'''import tensorflow as tf
from keras.layers import Layer


a = tf.constant([[0.1,0.2,0.3],[0.3,0.4,0.5]])
b = tf.constant([[0.5,0.6,0.7],[0.7,0.8,0.9]])

w=tf.constant([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])

similarity_matrix=tf.constant([])
print(similarity_matrix)

for i in range(2) :
    for j in range(2):
        c=tf.concat([a[i],b[j],tf.multiply(a[i],b[j])],0)
        print(c)
        alpha=tf.tensordot(w,c,1)
        alpha=tf.reshape(alpha,shape=(1,))
        similarity_matrix=tf.concat([similarity_matrix,alpha],0)

similarity_matrix=tf.reshape(similarity_matrix,shape=(2,2))
print(similarity_matrix)
print(tf.nn.softmax(similarity_matrix))'''
'''import tensorflow as tf

a=[[1,2],[3,4],[5,6]]

w=[1,1]

p=tf.tensordot(a,tf.transpose(w),1)

print(tf.nn.softmax(p))'''









'''class MyLayer1(Layer):
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
myLayer2(y)'''




from Word_embedding import W2VecLayer
from contexual_embedding import C2VecLayer
from BiDAF import BiAttentionLayer
from Modelling import ModellingLayer
from output import OutputLayer

w2vec=W2VecLayer(5)
x=w2vec.call(['i am dip','who i'])
c2vec=C2VecLayer()
y=c2vec(x)
bidaf = BiAttentionLayer()
z=bidaf(y)
modelling=ModellingLayer()
a=modelling(z)
output=OutputLayer()
b=output(a)

print(b)


