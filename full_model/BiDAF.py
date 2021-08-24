import tensorflow as tf
from keras.callbacks import *
from tensorflow.keras.layers import Input, LSTM, Bidirectional
from tensorflow.keras.models import Model, load_model
from layers.highway_layer import Highway
from layers.similarity_matrix import SimilarityMatrix
from layers.C2Q import C2Q_Layer
from layers.Q2C import Q2C_Layer
from layers.mega_merge import MegaMerge
from layers.modeling import ModellingLayer
from layers.output import OutputLayer


class BiDAF:
    def __init__(self, vector_dimension, max_context_length=None, max_query_length=None):
        self.vector_dimension = vector_dimension
        self.max_context_length = max_context_length
        self.max_query_length = max_query_length
        with tf.device('/device:GPU:0'):
            passage_input = Input(shape=(self.max_context_length, vector_dimension), dtype='float32', name="passage_input")
            question_input = Input(shape=(self.max_query_length, vector_dimension), dtype='float32', name="question_input")

            highway_layers = Highway(name='highway_layer')
            query_embedding = highway_layers(question_input)
            context_embedding = highway_layers(passage_input)

            contextual_embedding = Bidirectional(LSTM(vector_dimension, activation="tanh", recurrent_activation="sigmoid",
                                                return_sequences=True), name='bidirectional_encoder')
            contextual_embedded_query = contextual_embedding(query_embedding)
            contextual_embedded_context = contextual_embedding(context_embedding)

            similarity_matrix = SimilarityMatrix(name="sm")([contextual_embedded_context, contextual_embedded_query])
            c2q = C2Q_Layer(name="c2q")([similarity_matrix, contextual_embedded_query])
            q2c = Q2C_Layer(name="q2c")([similarity_matrix, contextual_embedded_context])
            megamerge = MegaMerge(name="mega")([contextual_embedded_context, c2q, q2c])
            t1, t2 = ModellingLayer(name="modelling")(megamerge)
            p1, p2 = OutputLayer(name="output")([t1, t2])

            model = Model(inputs=[passage_input, question_input], outputs=[p1, p2])
            model.summary()
            model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
            # for i, w in enumerate(model.weights): print(i, w.name)
            self.model = model

    def predict(self, c, q):
        p1, p2 = self.model.predict(x={"passage_input": c, "question_input": q}, batch_size=1)
        return p1, p2

    def train_model(self, x1, x2, y1, y2, epochs=1):
        with tf.device('/device:GPU:0'):
            filepath="drive/My Drive/app/bidaf_{epoch:03d}.h5"
            checkpoint = ModelCheckpoint(filepath, verbose=1)
            callbacks_list = [checkpoint]
            history = self.model.fit(x={"passage_input": x1, "question_input": x2}, y={"output": y1, "output_1": y2},
                                     batch_size=10,
                                     epochs=epochs, verbose=2,
                                     callbacks=callbacks_list)
            self.model.save('drive/My Drive/app/bidaf250_31.h5')
            return history, self.model

    def load_bidaf(self, path):
      custom_objects = {
          "Highway": Highway,
          "SimilarityMatrix": SimilarityMatrix,
          "C2Q_Layer": C2Q_Layer,
          "Q2C_Layer": Q2C_Layer,
          "MegaMerge": MegaMerge,
          "ModellingLayer": ModellingLayer,
          "OutputLayer": OutputLayer
      }

      self.model = load_model(path, custom_objects=custom_objects)