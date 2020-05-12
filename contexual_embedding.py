from keras.layers import Layer
from keras.layers import LSTM,Bidirectional

class C2VecLayer(Layer) :
    def __init__(self,**kwargs):
        super(C2VecLayer,self).__init__(**kwargs)

    def call(self,x):
        context,question=x[0],x[1]
        H=Bidirectional(LSTM(x[0].shape[2],
                             activation='sigmoid',
                             input_shape=(x[0].shape[1],x[0].shape[2]),
                             return_sequences=True))
        U=Bidirectional(LSTM(x[1].shape[2],
                             activation='sigmoid',
                             input_shape=(x[1].shape[1],x[1].shape[2]),
                             return_sequences=True))
        y=list()
        y.append(H)
        y.append(U)

        return y


'''
from glove_embedding import embedding
from tqdm import tqdm

from keras.models import Sequential
from keras.layers import LSTM, Bidirectional
from keras.layers import Dense
from keras.layers import TimeDistributed
import numpy

total_train_examples=86304
file_dir="F://Pycharm Projects//Spl2-nlp-QA//preproccess_squad//"
#according to "F://Pycharm Projects//Spl2-nlp-QA//preproccess_squad//max length.txt"
context_max_len=766 #train
question_max_len=60 #train

T=numpy.array([])

model1=Sequential()
model1.add(Bidirectional(LSTM(50,activation='sigmoid', input_shape=(766,50),return_sequences=True)))
model1.add(LSTM(100, activation='sigmoid', return_sequences=True))
model1.add(TimeDistributed(Dense(50)))
model1.compile(optimizer='adam', loss='categorical_crossentropy')

model2=Sequential()
model2.add(Bidirectional(LSTM(50,activation='sigmoid', input_shape=(60,50),return_sequences=True)))
model2.add(LSTM(100, activation='sigmoid', return_sequences=True))
model2.add(TimeDistributed(Dense(50)))
model2.compile(optimizer='adam', loss='categorical_crossentropy')

def train_model(is_context,is_question,file_name) :
    if is_context :
        with open(file_dir+file_name,encoding="utf8") as file :
            for line in tqdm(file,total=total_train_examples):
                T=numpy.array(embedding(line,context_max_len))
                T=T.reshape((1,context_max_len,50))
                model1.fit(T,T, epochs=1,verbose=0)

        contex_emb_model_json = model1.to_json()
        with open("contex_emb_model.json", "w") as json_file:
            json_file.write(contex_emb_model_json)
        model1.save_weights("contex_emb_model.h5")

    elif is_question :
        with open(file_dir+file_name,encoding="utf8") as file :
            for line in tqdm(file,total=total_train_examples):
                T=numpy.array(embedding(line,question_max_len))
                T=T.reshape((1,question_max_len,50))
                model2.fit(T,T, epochs=1,verbose=0)

        question_emb_model_json = model2.to_json()
        with open("question_emb_model.json", "w") as json_file:
            json_file.write(question_emb_model_json)
        model2.save_weights("question_emb_model.h5")

    print("Saved model to disk")
'''