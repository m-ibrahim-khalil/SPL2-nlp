

'''import tensorflow as tf
from keras import Sequential

from Word_embedding import W2VecLayer
from contexual_embedding import C2VecLayer
from BiDAF import BiAttentionLayer
from Modelling import ModellingLayer
from output import OutputLayer
import get_contextual_layer_inputs as getIn


def custom_loss_func(y_actual,y_predicted) :
    p1_actual=y_actual[0]
    p1_predicted=y_predicted[0]

    p2_actual=y_actual[1]
    p2_predicted=y_predicted[1]

    cce=tf.keras.losses.CategoricalCrossentropy()

    return cce(p1_actual,p1_predicted) + cce(p2_actual,p2_predicted)




#word2vec=W2VecLayer(3)
#input=word2vec.call(['i am dip','who i'])
#output=tf.constant([[0,0,1],[0,0,1]], dtype=tf.float32)

model=Sequential()
model.add(C2VecLayer())
model.add(BiAttentionLayer())
model.add(ModellingLayer())
model.add(OutputLayer())
model.compile(loss=custom_loss_func , optimizer='adam')

y=model(getIn(['i am dip','who i']))
model.summary()

#model.fit(x=getIn(['i am dip','who i']), y=output, steps_per_epoch=1, epochs=5)
'''

import get_contextual_layer_inputs as getIn
from BiDAF import BiAttentionLayer
from contexual_embedding import C2VecLayer
from Modelling import ModellingLayer
from output import OutputLayer
import tensorflow as tf
from keras.models import Sequential

# passage_input = keras.layers.Input(shape=(None, 400), dtype='float32', name="passage_input")
# question_input = keras.layers.Input(shape=(None, 400), dtype='float32', name="question_input")
# y = getIn.get_contextual_inputs(input)

def custom_loss_func(y_actual,y_predicted) :
    p1_actual=y_actual[0]
    p1_predicted=y_predicted[0]

    p2_actual=y_actual[1]
    p2_predicted=y_predicted[1]

    cce=tf.keras.losses.CategoricalCrossentropy()

    return cce(p1_actual,p1_predicted) + cce(p2_actual,p2_predicted)



inputs = ['Ibrahim khalil is a good boy', 'who is ibrahim ?']

model = Sequential()

model.add(C2VecLayer())
model.add(BiAttentionLayer())
model.add(ModellingLayer())
model.add(OutputLayer())
model.compile(loss=custom_loss_func,optimizer='adam')

output=tf.zeros(shape=(2,766),dtype=tf.float32)

model.fit(x=getIn.get_contextual_inputs(inputs),y=output,steps_per_epoch=1)

model.summary()

