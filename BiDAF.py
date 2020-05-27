#dynamic

from keras.layers import Layer
import tensorflow as tf
from keras import backend as K
from keras.layers.advanced_activations import Softmax


class BiAttentionLayer(Layer):

    def __init__(self, **kwargs):
        super(BiAttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.shape=input_shape
        print("dukche 2 build")

        self.kernel = self.add_weight(name='kernel',
                                      shape=(3 * input_shape[3], 1),
                                      initializer='uniform',
                                      trainable=True)

        super(BiAttentionLayer, self).build(input_shape)

    def compute_similarity(self, repeated_context_vectors, repeated_query_vectors):
        print("dukche 2 csm")
        element_wise_multiply = repeated_context_vectors * repeated_query_vectors
        concatenated_tensor = K.concatenate(
            [repeated_context_vectors, repeated_query_vectors, element_wise_multiply], axis=-1)
        print(concatenated_tensor.shape)
        dot_product = K.squeeze(K.dot(concatenated_tensor, self.kernel), axis=-1)
        print(dot_product.shape)
        print(dot_product)
        return dot_product

    @tf.function
    def build_similarity_matrix(self, context, question):
        print("dukche 2 sm")
        num_context_words = K.shape(context)[1]
        print(num_context_words)
        num_query_words = K.shape(question)[1]
        context_dim_repeat = K.concatenate([[1, 1], [num_query_words], [1]], 0)
        print(context_dim_repeat)
        query_dim_repeat = K.concatenate([[1], [num_context_words], [1, 1]], 0)
        repeated_context_vectors = K.tile(K.expand_dims(context, axis=2), context_dim_repeat)
        print(repeated_context_vectors)
        repeated_query_vectors = K.tile(K.expand_dims(question, axis=1), query_dim_repeat)
        print(repeated_query_vectors)
        similarity_matrix = self.compute_similarity(repeated_context_vectors, repeated_query_vectors)
        print(similarity_matrix)
        return similarity_matrix
        '''
        count1 = 0
        count2 = 0
        similarity_matrix = tf.constant([])
        for i in range(self.shape[2]):
            count1 += 1
            for j in range(self.shape[2]):
                count2 += 1
                c = tf.concat([context[i], question[j], tf.multiply(context[i], question[j])], 0)
                alpha = tf.tensordot(self.kernel, c, 1)
                alpha = tf.reshape(alpha, shape=(1,))
                similarity_matrix = tf.concat([similarity_matrix, alpha], 0)
                print(count1)
                print(count2)

        similarity_matrix = tf.reshape(similarity_matrix, shape=(self.shape[2], self.shape[2]))
        return similarity_matrix

        '''

    @tf.function
    def C2Q_Attention(self, question):
        context_to_query_attention = Softmax(axis=-1)(self.similarity_matrix)
        print("dukche 2 c2q")
        encoded_question = K.expand_dims(question, axis=1)
        return K.sum(K.expand_dims(context_to_query_attention, axis=-1) * encoded_question, -2)
        '''
        U_A = tf.constant([])
        
        for j in range(self.shape[2]):
            temp = tf.zeros(shape=(self.shape[3],), dtype=tf.float32)
            for i in range(self.shape[2]):
                c = tf.tensordot(self.attention[j][i], question[i], 0)
                temp += c

            U_A = tf.concat([U_A, temp], 0)

        U_A = tf.reshape(U_A, shape=(self.shape[2], self.shape[3]))

        return U_A

        '''

    @tf.function
    def Q2C_Attention(self, context):

        b = tf.reduce_max(self.attention, axis=1, )
        print("dukche 2 q2c")

        temp = tf.zeros(shape=(self.shape[3],), dtype=tf.float32)

        for i in range(self.shape[2]):
            temp += tf.tensordot(b[i], context[i], 0)

        H_A = tf.constant([])

        for i in range(self.shape[2]):
            H_A = tf.concat([H_A, temp], 0)

        return tf.reshape(H_A, shape=(self.shape[2], self.shape[3]))

    def megamerge(self, context, U_A, H_A):
        G = tf.constant([])
        print("dukche 2 mm")
        for t in range(self.shape[2]):
            temp = tf.concat([context[t], U_A[t], tf.multiply(context[t], U_A[t]), tf.multiply(context[t], H_A[t])], 0)
            G = tf.concat([G, temp], 0)
            # G.append(np.concatenate((context[t],U_A[t],np.multiply(context[t],U_A[t]),np.multiply(context[t],H_A[t]))))
        return tf.reshape(G, shape=(self.shape[2], self.shape[3]*4))

    def call(self, x):
        print("dukche 2")
        context, question = x[0][0], x[1][0]
        context1,question1 = x[0], x[1]
        self.similarity_matrix = self.build_similarity_matrix(context1, question1)
        self.attention = tf.nn.softmax(self.similarity_matrix, axis=-1)
        self.U_A = self.C2Q_Attention(question)
        self.H_A = self.Q2C_Attention(context)
        self.G = self.megamerge(context, self.U_A, self.H_A)

        return self.G

    def compute_output_shape(self, input_shape):
        print("dukche 2")
        return (self.shape[2], self.shape[3]*4)
