from keras.layers import Layer
import tensorflow as tf
from keras import backend as K
from keras.layers.advanced_activations import Softmax


class SimilarityMatrix(Layer):
    def __init__(self, **kwargs):
        super(SimilarityMatrix, self).__init__(**kwargs)

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
        similarity = K.squeeze(K.dot(concatenated_tensor, self.kernel), axis=-1)

        return similarity

    def build_similarity_matrix(self, context, question):
        num_context_words = K.shape(context)[1]
        num_query_words = K.shape(question)[1]
        context_dim_repeat = K.concatenate([[1, 1], [num_query_words], [1]], 0)
        query_dim_repeat = K.concatenate([[1], [num_context_words], [1, 1]], 0)
        context_vectors = K.tile(K.expand_dims(context, axis=2), context_dim_repeat)
        query_vectors = K.tile(K.expand_dims(question, axis=1), query_dim_repeat)
        similarity_matrix = self.calculate_similarity(context_vectors, query_vectors)
        return similarity_matrix

    def call(self, x):
        context, question = x
        self.similarity_matrix = self.build_similarity_matrix(context, question)
        return self.similarity_matrix

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], input_shape[1][1])