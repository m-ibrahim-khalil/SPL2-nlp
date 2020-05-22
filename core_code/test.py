import core_code.word_char_embd as wce
import core_code.word_embedding as we
import core_code.highway_layer as hwl
import numpy
from contexual_embedding import C2VecLayer
from tensorflow import keras
import tensorflow as tf
import keras.backend as K

directory = "F://Pycharm Projects//glove.6B.50d.txt"
output = we.get_glove(directory, 50)
emb_matrix = output[0]
word2id = output[1]
id2word = output[2]
word_dict = word2id
char_dict, _, _ = wce.create_char_dicts()

inputs, embd_layer = wce.get_embedding_layer(
        word_dict_len=len(word_dict),
        char_dict_len=len(char_dict),
        max_word_len=16,
        word_embd_dim=50,
        char_embd_dim=30,
        char_hidden_dim=30,
        char_hidden_layer_type='cnn',
        word_embd_weights=emb_matrix
    )
model1 = keras.models.Model(inputs=inputs, outputs=embd_layer)
sentences = [
        ['All', 'work', 'and', 'no', 'play']
]
in_ = wce.get_batch_input(sentences,word_dict,char_dict)
z = model1(in_)

question = [
        'who is jack ma ?'
]

question = [question[0].split()]
print(question)

q_in = wce.get_batch_input(question, word_dict,char_dict)
q_emb = model1(q_in)

print(z)
print(q_emb)

highway_layer = hwl.Highway(transform_gate_bias=-2)

H = highway_layer(z)
U = highway_layer(q_emb)

H = tf.cast(H, dtype=tf.int32)
U = tf.cast(U, dtype=tf.int32)

y = tf.concat([H, U], 0)
y = tf.reshape(y,[2,1,5,80])

print(H)
print(U)
print(y)


contextualLayer = C2VecLayer()
y = contextualLayer(y)
print(y)

