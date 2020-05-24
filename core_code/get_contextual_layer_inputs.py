import core_code.word_char_embd as wce
import core_code.word_embedding as we
import core_code.highway_layer as hwl
from tensorflow import keras
import tensorflow as tf

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


def get_contextual_inputs(input):
    highway_layer = hwl.Highway(transform_gate_bias=-2)
    context = [input[0].split()]
    question = [input[1].split()]
    context = wce.get_batch_input(context, word_dict, char_dict)
    question = wce.get_batch_input(question, word_dict, char_dict)
    context_embeding = model1(context)
    question_embeding = model1(question)
    H = highway_layer(context_embeding)
    U = highway_layer(question_embeding)
    # print(H)
    padcon = tf.constant(0,tf.float32,[1,765-len(context),80])
    padqn = tf.constant(0,tf.float32,[1,765-len(question),80])
    # print(padcon)
    # print(len(context))
    # print(len(question))
    # print(padqn)
    H = tf.concat([H,padcon],1)
    U = tf.concat([U,padqn],1)
    # print(H)
    # print(U)
    y = tf.concat([H, U], 0)
    # print(y)
    y = tf.reshape(y, [2, 1, 766, 80])
    return y

