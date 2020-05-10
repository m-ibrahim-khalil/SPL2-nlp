from vocab import get_glove
import numpy as np

emb_matrix, word2id, id2word=get_glove("F://Pycharm Projects//Spl2-nlp-QA//glove.6B.50d.txt",50)

glove_dim=50

def embedding(line,max_len):
    T = np.array([])

    for word in line.split():
        if word not in word2id.keys():
            T=np.append(T,np.random.rand(glove_dim))
            continue
        id=word2id[word]
        T=np.append(T,np.array(emb_matrix[id]))

    append=np.zeros(((max_len-len(line.split())),50),dtype=float) #question er jonno 60
    T=np.append(T,append)

    return T
