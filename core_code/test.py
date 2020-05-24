import core_code.get_contextual_layer_inputs as getIn
from contexual_embedding import C2VecLayer

input = ["I am Ibrahim", "Who i am "]
y = getIn.get_contextual_inputs(input)
print(y)
contextualLayer = C2VecLayer()
y = contextualLayer(y)
print(y)