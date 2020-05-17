from keras.layers import Layer
class OutputLayer(Layer):
    def __init__(self,**kwargs):
        super(OutputLayer, self).__init__(**kwargs)

    def build(self):
        self.w1=self.add_weight(name='w1',
                                shape=(10*50,),
                                initializer='uniform',
                                trainable=True)
        self.w2=self.add_weight(name='w2',
                                shape=(10*50,),
                                initializer='uniform',
                                trainable=True)

        super(OutputLayer, self).build()

    def call(self, x):
