from keras.layers import Layer
import numpy as np

class BiAttentionLayer(Layer) :

    def __init__(self,**kwargs):
        super(BiAttentionLayer,self).__init__(**kwargs)

    def build(self,input_shape):
        self.kernel=self.add_weight(name='kernel',
                                    shape=(6*50,),
                                    initializer='uniform',
                                    trainable=True)
        super(BiAttentionLayer, self).build(input_shape)

    def softmax(self,x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def attention_distribution(self,similarity_matrix):
        A=list()
        for row in similarity_matrix:
            A.append(self.softmax(row))
        return A

    def build_similarity_matrix(self,context,question):
        similarity_matrix = list()
        row = 0
        for a in context:
            column = 0
            for b in question:
                c = np.concatenate((a, b, np.multiply(a, b)))
                alpha = np.dot(self.kernel, c)
                similarity_matrix[row][column] = alpha
                column += 1
            row += 1

        return similarity_matrix

    def C2Q_Attention(self,question):
        U_A=list()

        for row in self.attention:
            t=0
            temp=[0 for i in range(len(question[t]))]
            for element in row :
                a=np.dot(element,question[t])
                temp=[c+d for c,d in zip(temp,a)]
                t+=1
            U_A.append(temp)

        return U_A

    def Q2C_Attention(self,context):
        H_A=list()
        z=list()

        for row in self.similarity_matrix :
            z.append(max(row))

        b=self.softmax(z)

        temp=[0 for i in range(context.shape[0])]
        t=0
        for element in b:
            a=np.dot(element,context[t])
            temp = [c + d for c, d in zip(temp, a)]
            t+=1

        for i in range(len(context)) :
            H_A.append(temp)

        return H_A

    def megamerge(self,context,U_A,H_A):
        G=list()
        for t in range(len(context)):
            G.append(np.concatenate((context[t],U_A[t],np.multiply(context[t],U_A[t]),np.multiply(context[t],H_A[t]))))
        return G

    def call(self,x):

        context, question = x[0][0], x[1][0]
        self.similarity_matrix=self.build_similarity_matrix(context,question)
        self.attention=self.attention_distribution(self.similarity_matrix)
        self.U_A=self.C2Q_Attention(question)
        self.H_A=self.Q2C_Attention(context)
        self.G=self.megamerge(context,self.U_A,self.H_A)
