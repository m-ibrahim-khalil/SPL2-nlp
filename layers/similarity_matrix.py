"""
    Similarity matrix inputs the contextual embedding of (context and query)
    Find the similarity between the context and query
    output a (context_length, query_length) matrix
"""

from keras.layers import Layer
import tensorflow as tf
from keras import backend as k


class SimilarityMatrix(Layer):
    def __init__(self, **kwargs):
        super(SimilarityMatrix, self).__init__(**kwargs)
        self.context_shape = None
        self.question_shape = None
        self.kernel = None
        self.similarity_matrix = None

    def build(self, input_shape):
        self.context_shape = input_shape[0]
        self.question_shape = input_shape[1]

        self.kernel = self.add_weight(name="kernel",
                                      shape=(3 * input_shape[0][2], 1),
                                      initializer='uniform',
                                      trainable=True)

        super(SimilarityMatrix, self).build(input_shape)

    def calculate_similarity(self, context_vectors, query_vectors):
        element_wise_multiply = context_vectors * query_vectors
        concatenated_tensor = tf.concat(
            [context_vectors, query_vectors, element_wise_multiply], axis=-1)
        similarity = k.squeeze(k.dot(concatenated_tensor, self.kernel), axis=-1)

        return similarity

    def build_similarity_matrix(self, context, question):
        num_context_words = k.shape(context)[1]
        num_query_words = k.shape(question)[1]
        context_dim_repeat = k.concatenate([[1, 1], [num_query_words], [1]], 0)
        query_dim_repeat = k.concatenate([[1], [num_context_words], [1, 1]], 0)
        context_vectors = k.tile(k.expand_dims(context, axis=2), context_dim_repeat)
        query_vectors = k.tile(k.expand_dims(question, axis=1), query_dim_repeat)
        similarity_matrix = self.calculate_similarity(context_vectors, query_vectors)
        return similarity_matrix

    def call(self, x):
        context, question = x
        self.similarity_matrix = self.build_similarity_matrix(context, question)
        return self.similarity_matrix

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][1], input_shape[1][1]
