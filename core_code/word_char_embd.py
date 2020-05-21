import numpy
from tensorflow import keras
import keras.backend as K
import core_code.word_embedding as we

__all__ = [
    'MaskedConv1D', 'MaskedFlatten',
    'get_batch_input', 'get_embedding_layer',
]


class MaskedConv1D(keras.layers.Conv1D):

    def __init__(self, **kwargs):
        super(MaskedConv1D, self).__init__(**kwargs)
        self.supports_masking = True

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            inputs *= K.expand_dims(mask, axis=-1)
        return super(MaskedConv1D, self).call(inputs)


class MaskedFlatten(keras.layers.Flatten):

    def __init__(self, **kwargs):
        super(MaskedFlatten, self).__init__(**kwargs)
        self.supports_masking = True

    def compute_mask(self, inputs, mask=None):
        return mask


def get_batch_input(sentences,
                    word_dict,
                    char_dict,
                    max_word_len=16,
                    word_unknown=1,
                    char_unknown=1,
                    word_ignore_case=True,
                    char_ignore_case=True):
    """Convert sentences to desired input tensors.
    :param sentences: A list of lists representing the input sentences.
    :param max_word_len: The maximum allowed length of word.
    :param word_dict: Map a word to an integer. (0 and 1 should be preserved)
    :param char_dict: Map a character to an integer. (0 and 1 should be preserved)
    :param word_unknown: An integer representing the unknown word.
    :param char_unknown: An integer representing the unknown character.
    :param word_ignore_case: Word will be transformed to lower case before mapping.
    :param char_ignore_case: Character will be transformed to lower case before mapping.
    :return word_embd_input, char_embd_input: The desired inputs.
    """
    sentence_num = len(sentences)

    max_sentence_len = max(map(len, sentences))

    word_embd_input = [[0] * max_sentence_len for _ in range(sentence_num)]
    char_embd_input = [[[0] * max_word_len for _ in range(max_sentence_len)] for _ in range(sentence_num)]

    for sentence_index, sentence in enumerate(sentences):
        for word_index, word in enumerate(sentence):
            if word_ignore_case:
                word_key = word.lower()
            else:
                word_key = word
            word_embd_input[sentence_index][word_index] = word_dict.get(word_key, word_unknown)
            for char_index, char in enumerate(word):
                if char_index >= max_word_len:
                    break
                if char_ignore_case:
                    char = char.lower()
                char_embd_input[sentence_index][word_index][char_index] = char_dict.get(char, char_unknown)

    return [numpy.asarray(word_embd_input), numpy.asarray(char_embd_input)]


def get_embedding_layer(word_dict_len,
                        char_dict_len,
                        max_word_len,
                        word_embd_dim=50,
                        char_embd_dim=30,
                        char_hidden_dim=150,
                        char_hidden_layer_type='lstm',
                        word_embd_weights=None,
                        char_embd_weights=None,
                        word_embd_trainable=False,
                        char_embd_trainable=None,
                        word_mask_zero=True,
                        char_mask_zero=True):
    """Get the merged embedding layer.
    :param word_dict_len: The number of words in the dictionary including the ones mapped to 0 or 1.
    :param char_dict_len: The number of characters in the dictionary including the ones mapped to 0 or 1.
    :param max_word_len: The maximum allowed length of word.
    :param word_embd_dim: The dimensions of the word embedding.
    :param char_embd_dim: The dimensions of the character embedding
    :param char_hidden_dim: The dimensions of the hidden states of RNN in one direction.
    :param word_embd_weights: A numpy array representing the pre-trained embeddings for words.
    :param char_embd_weights: A numpy array representing the pre-trained embeddings for characters.
    :param word_embd_trainable: Whether the word embedding layer is trainable.
    :param char_embd_trainable: Whether the character embedding layer is trainable.
    :param char_hidden_layer_type: The type of the recurrent layer, 'lstm' or 'gru'.
    :param word_mask_zero: Whether enable the mask for words.
    :param char_mask_zero: Whether enable the mask for characters.
    :return inputs, embd_layer: The keras layer.
    """
    if word_embd_weights is not None:
        word_embd_weights = [word_embd_weights]
    if word_embd_trainable is None:
        word_embd_trainable = word_embd_weights is None

    if char_embd_weights is not None:
        char_embd_weights = [char_embd_weights]
    if char_embd_trainable is None:
        char_embd_trainable = char_embd_weights is None

    word_input_layer = keras.layers.Input(
        shape=(None,),
        name='Input_Word',
    )
    char_input_layer = keras.layers.Input(
        shape=(None, max_word_len),
        name='Input_Char',
    )

    word_embd_layer = keras.layers.Embedding(
        input_dim=word_dict_len,
        output_dim=word_embd_dim,
        mask_zero=word_mask_zero,
        weights=word_embd_weights,
        trainable=word_embd_trainable,
        name='Embedding_Word',
    )(word_input_layer)
    char_embd_layer = keras.layers.Embedding(
        input_dim=char_dict_len,
        output_dim=char_embd_dim,
        mask_zero=char_mask_zero,
        weights=char_embd_weights,
        trainable=char_embd_trainable,
        name='Embedding_Char_Pre',
    )(char_input_layer)
    if char_hidden_layer_type == 'lstm':
        char_hidden_layer = keras.layers.Bidirectional(
            keras.layers.LSTM(
                units=char_hidden_dim,
                input_shape=(max_word_len, char_dict_len),
                return_sequences=False,
                return_state=False,
            ),
            name='Bi-LSTM_Char',
        )
    elif char_hidden_layer_type == 'cnn':
        char_hidden_layer = [
            MaskedConv1D(
                filters=max(1, char_hidden_dim // 5),
                kernel_size=3,
                activation='relu',
            ),
            MaskedFlatten(),
            keras.layers.Dense(
                units=char_hidden_dim,
                name='Dense_Char',
            ),
        ]
    elif isinstance(char_hidden_layer_type, list) or isinstance(char_hidden_layer_type, keras.layers.Layer):
        char_hidden_layer = char_hidden_layer_type
    else:
        raise NotImplementedError('Unknown character hidden layer type: %s' % char_hidden_layer_type)
    if not isinstance(char_hidden_layer, list):
        char_hidden_layer = [char_hidden_layer]
    for i, layer in enumerate(char_hidden_layer):
        if i == len(char_hidden_layer) - 1:
            name = 'Embedding_Char'
        else:
            name = 'Embedding_Char_Pre_%d' % (i + 1)
        char_embd_layer = keras.layers.TimeDistributed(layer=layer, name=name)(char_embd_layer)
    embd_layer = keras.layers.Concatenate(
        name='Embedding',
    )([word_embd_layer, char_embd_layer])
    return [word_input_layer, char_input_layer], embd_layer


def create_char_dicts(CHAR_PAD_ID=0, CHAR_UNK_ID=1, _CHAR_PAD='*', _CHAR_UNK='$'):
    unique_chars = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '+', ',', '-', '.', '/', '0', '1', '2', '3',
                    '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '[', ']', '^', 'a', 'b', 'c', 'd',
                    'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
                    'x', 'y', 'z', '~', ]  # based on analysis in jupyter notebook

    num_chars = len(unique_chars)

    idx2char = dict(enumerate(unique_chars, 2))
    idx2char[CHAR_PAD_ID] = _CHAR_PAD
    idx2char[CHAR_UNK_ID] = _CHAR_UNK
    char2idx = {v: k for k, v in idx2char.items()}
    return char2idx, idx2char, num_chars


if __name__ == '__main__':
    directory = "F://Pycharm Projects//glove.6B.50d.txt"
    output = we.get_glove(directory, 50)
    emb_matrix = output[0]
    word2id = output[1]
    id2word = output[2]
    word_dict = word2id
    char_dict, _, _ = create_char_dicts()

    inputs, embd_layer = get_embedding_layer(
        word_dict_len=len(word_dict),
        char_dict_len=len(char_dict),
        max_word_len=16,
        word_embd_dim=50,
        char_embd_dim=30,
        char_hidden_dim=30,
        char_hidden_layer_type='cnn',
        word_embd_weights=emb_matrix
    )

    sentences = [
        ['All', 'work', 'and', 'no', 'play'],
        ['makes', 'Jack', 'a', 'dull', 'boy', '.'],
    ]
    # wc_embd = WordCharEmbd(
    #     word_min_freq=0,
    #     char_min_freq=0,
    #     word_ignore_case=False,
    #     char_ignore_case=False,
    # )
    # for sentence in sentences:
    #     wc_embd.update_dicts(sentence)

    model = keras.models.Model(inputs=inputs, outputs=embd_layer)
    model.compile(
        optimizer='adam',
        loss=keras.losses.sparse_categorical_crossentropy,
        metrics=[keras.metrics.sparse_categorical_accuracy],
    )
    model.summary()

    in_ = get_batch_input(sentences,word_dict,char_dict)
    print(in_[0])
    print(in_[1])

    y = model(in_)

    print(y)



