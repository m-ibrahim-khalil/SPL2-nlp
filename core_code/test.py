import core_code.get_contextual_layer_inputs as getIn
from contexual_embedding import C2VecLayer
import tensorflow as tf
import keras
inputs = ['Imrahim khalil is a good boy', 'who is ibrahim ?']
# passage_input = keras.layers.Input(shape=(766, 400), dtype='float32', name="passage_input")
# question_input = keras.layers.Input(shape=(766, 400), dtype='float32', name="question_input")
y = getIn.get_contextual_inputs(inputs)
print(y)
contextualLayer = C2VecLayer()
y = contextualLayer(y)
print(y)
