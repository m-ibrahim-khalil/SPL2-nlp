import core_code.word_char_embd as wce
import core_code.word_embedding as we
import core_code.highway_layer as hwl
import numpy
import keras
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
        char_hidden_dim=100,
        char_hidden_layer_type='cnn',
        word_embd_weights=emb_matrix
    )

highway_layer = hwl.Highway()(embd_layer)

model = keras.models.Model(inputs=inputs, outputs=highway_layer)

model.summary()