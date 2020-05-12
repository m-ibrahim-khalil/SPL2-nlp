import tensorflow as tf
from keras.layers import Layer
from tensorflow.python.ops import embedding_ops


class CharEmbeddingLayer(Layer):

    def __init__(self, **kwargs):
        self.char_embedding_size = 8
        self.word_max_len = 16
        self.char_out_size = 100 #same as filter size
        self.window_width = 5
        self.context_len = 300
        self.question_len = 30
        self.keep_prob = .5
        _, _, num_chars = self.create_char_dicts()
        self.char_vocab = num_chars
        super(CharEmbeddingLayer, self).__init__(**kwargs)

    def conv1d(self, input_, output_size, width, stride):
        input_size = input_.get_shape()[-1]
        input_ = tf.expand_dims(input_, axis=1)
        filter_ = tf.get_variable("conv_filter", shape=[1, width, input_size, output_size])
        convolved = tf.nn.conv2d(input_, filter=filter_, strides=[1, 1, stride, 1], padding="VALID")
        result = tf.squeeze(convolved, axis=1)
        return result

    def call(self, char_ids_context, char_ids_qn):
        char_emb_matrix = tf.Variable(
            tf.random_uniform((self.char_vocab, self.char_embedding_size), -1, 1))  # is trainable

        print("Shape context placeholder", char_ids_context.shape)
        print("Shape qn placeholder", char_ids_qn.shape)

        self.context_char_embs = embedding_ops.embedding_lookup(char_emb_matrix, tf.reshape(char_ids_context,shape=(-1,self.word_max_len)))
        self.context_char_embs = tf.reshape(self.context_char_embs, shape=(
            -1, self.word_max_len,
            self.char_embedding_size))
        print("Shape context embs before conv", self.context_char_embs.shape)

        self.qn_char_embs = embedding_ops.embedding_lookup(char_emb_matrix,
                                                           tf.reshape(char_ids_qn, shape=(-1, self.word_max_len)))
        self.qn_char_embs = tf.reshape(self.qn_char_embs,
                                       shape=(-1, self.word_max_len, self.char_embedding_size))

        print("Shape qn embs before conv", self.qn_char_embs.shape)
        self.context_emb_out = self.conv1d(input_=self.context_char_embs, output_size=self.char_out_size,
                                      width=self.window_width, stride=1)

        self.context_emb_out = tf.nn.dropout(self.context_emb_out, self.keep_prob)

        print("Shape context embs after conv", self.context_emb_out.shape)

        self.context_emb_out = tf.reduce_sum(self.context_emb_out, axis=1)

        self.context_emb_out = tf.reshape(self.context_emb_out, shape=(-1, self.context_len,
                                                                       self.char_out_size))
        print("Shape context embs after pooling", self.context_emb_out.shape)

        self.qn_emb_out = self.conv1d(input_=self.qn_char_embs, output_size=self.char_out_size,
                                 width=self.window_width, stride=1)

        self.qn_emb_out = tf.nn.dropout(self.qn_emb_out, self.keep_prob)

        print("Shape qn embs after conv", self.qn_emb_out.shape)

        self.qn_emb_out = tf.reduce_sum(self.qn_emb_out,
                                        axis=1)

        self.qn_emb_out = tf.reshape(self.qn_emb_out, shape=(-1, self.question_len, self.char_out_size))
        print("Shape qn embs after pooling", self.qn_emb_out.shape)

        return self.context_emb_out, self.qn_emb_out

    def create_char_dicts(self, CHAR_PAD_ID=0, CHAR_UNK_ID=1, _CHAR_PAD='*', _CHAR_UNK='$'):
        unique_chars = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '+', ',', '-', '.', '/', '0', '1', '2', '3',
                        '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '[', ']', '^', 'a', 'b', 'c', 'd',
                        'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
                        'x', 'y', 'z', '~', ]  # based on analysis in jupyter notebook

        num_chars = len(unique_chars)

        idx2char = dict(enumerate(unique_chars, 2))
        idx2char[CHAR_PAD_ID] = _CHAR_PAD
        idx2char[CHAR_UNK_ID] = _CHAR_UNK
        char2idx = {v: k for k, v in idx2char.iteritems()}
        return char2idx, idx2char, num_chars
