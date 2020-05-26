import core_code.get_contextual_layer_inputs as getIn
from BiDAF import BiAttentionLayer
from contexual_embedding import C2VecLayer
from Modelling import ModellingLayer
from output import OutputLayer
import tensorflow as tf
from keras.models import Sequential

# passage_input = keras.layers.Input(shape=(None, 400), dtype='float32', name="passage_input")
# question_input = keras.layers.Input(shape=(None, 400), dtype='float32', name="question_input")
# y = getIn.get_contextual_inputs(input)

inputs = ['Imrahim khalil is a good boy', 'who is ibrahim ?']

model = Sequential()

model.add(C2VecLayer())
model.add(BiAttentionLayer())
model.add(ModellingLayer())
model.add(OutputLayer())

model(getIn.get_contextual_inputs(inputs))

model.summary()



