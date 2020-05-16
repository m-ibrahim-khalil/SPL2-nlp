'''import tensorflow as tf

a=tf.Variable(tf.zeros([5,5]))
a[2,3]=5
print(a)'''



'''import tensorflow as tf
from keras.layers import Layer
class faltuLayer(Layer):
    def __init__(self,**kwargs):
        super(faltuLayer,self).__init__(**kwargs)

    def call(self, x):
        print(tf.keras.backend.get_value(x))
        return x

class faltuLayer2(Layer):
    def __init__(self,**kwargs):
        super(faltuLayer2,self).__init__(**kwargs)

    def call(self, x):
        with tf.compat.v1.Session() as sess:
            print(sess.run(x))
        return x

x=tf.Variable(tf.zeros([500,500]))
print(x)
fal=faltuLayer()
fal2=faltuLayer2()
y=fal(x)
fal2(y)'''



from Word_embedding import W2VecLayer
from contexual_embedding import C2VecLayer
from BiDAF import BiAttentionLayer

w2vec=W2VecLayer(5)
x=w2vec.call(['i am dip','who i'])
c2vec=C2VecLayer()
y=c2vec(x)
bidaf = BiAttentionLayer()
z=bidaf(y)

