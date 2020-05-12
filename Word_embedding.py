from vocab import get_glove
from keras.layers import Layer
import numpy as np

emb_matrix, word2id, id2word=get_glove("F://Pycharm Projects//Spl2-nlp-QA//glove.6B.50d.txt",50)
glove_dim=50

class W2VecLayer(Layer):

    def __init__(self,context_max_len,question_max_len,**kwargs):
        self.context_max_len=context_max_len
        self.question_max_len=question_max_len

        super(W2VecLayer,self).__init__(**kwargs)

    def build(self):
        self.kernel=self.add_weight(name='kernel',
                                    trainable=False)

        super(W2VecLayer, self).build()

    def embedding(self,line,max_len):
        T=np.array([])
        for word in line.split() :
            if word not in word2id.keys():
                T = np.append(T, np.random.rand(glove_dim))
                continue
            id = word2id[word]
            T = np.append(T, np.array(emb_matrix[id]))

        append = np.zeros(((max_len - len(line.split())), glove_dim), dtype=float)
        T = np.append(T, append)
        T=T.reshape((1,max_len,glove_dim))
        return T


    def call(self,context,question):
        self.T=self.embedding(context,self.context_max_len)
        self.J=self.embedding(question,self.question_max_len)

        x=list()
        x.append(self.T)
        x.append(self.J)

        return x