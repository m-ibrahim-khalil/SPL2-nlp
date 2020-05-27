from keras.layers import Layer
import tensorflow as tf
from keras import backend as K
from keras.layers.advanced_activations import Softmax

# kernel = tf.constant([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12]])
kernel = tf.constant([[1],[1],[1],[1],[1],[1]])
print(kernel.shape)

def compute_similarity(repeated_context_vectors, repeated_query_vectors):
    print("dukche 2 csm")
    element_wise_multiply = repeated_context_vectors * repeated_query_vectors
    concatenated_tensor = K.concatenate(
        [repeated_context_vectors, repeated_query_vectors, element_wise_multiply], axis=-1)
    print(concatenated_tensor)
    dot_product = K.squeeze(K.dot(concatenated_tensor, kernel), axis=-1)
    print(dot_product.shape)
    print(dot_product)
    return dot_product


def build_similarity_matrix(context, question):
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
    similarity_matrix = compute_similarity(repeated_context_vectors, repeated_query_vectors)
    print(similarity_matrix)
    return similarity_matrix


context = tf.constant([[[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]]])
query = tf.constant([[[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]]])
context1 = tf.constant([[[1,2],[3,4]]])
query1 =tf.constant([[[5,6],[7,8]]])

print(context1.shape)
# query = tf.constant(1, tf.float32, [1, 766, 160])
sm = build_similarity_matrix(context1, query1)
sess = tf.compat.v1.Session()

with sess.as_default():
  assert tf.compat.v1.get_default_session() is sess
  print(sm.eval())



def c2q():
    context_to_query_attention = Softmax(axis=-1)(self.similarity_matrix)
    print("dukche 2 c2q")
    encoded_question = K.expand_dims(question, axis=1)
    return K.sum(K.expand_dims(context_to_query_attention, axis=-1) * encoded_question, -2)