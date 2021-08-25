import numpy as np
from layers.vectors import Vectors
from tqdm import tqdm
import tensorflow as tf

dir1 = "F://SPL2//preproccess_squad//train_set//"
dir = "F://SPL2//preproccess_squad//"

with tf.device('/device:GPU:0'):
    context_data = []
    question_data = []
    output_data1 = []
    output_data2 = []
    j = 0
    i = 0
    sample_size = 82096
    data_batch = 5864

    vectors = Vectors().load_vectors()
    with open(dir1+"250//train250.context", 'r', encoding='utf8') as context_file, \
            open(dir1+"250//train250.question", 'r', encoding='utf8') as question_file, \
            open(dir1+"250//train250.span", 'r', encoding='utf8') as span_file:

        for context, question, span in zip(tqdm(context_file, total=sample_size),
                                           tqdm(question_file, total=sample_size),
                                           tqdm(span_file, total=sample_size)):

            context = [context]
            question = [question]
            passage = [pas.strip() for pas in context]
            cont = []
            for pas in passage:
                context_tokens = pas.split(" ")
                cont.append(context_tokens)
            original_passage = [pas.lower() for pas in passage]
            quest = []
            for ques in question:
                question_tokens = ques.split(" ")
                quest.append(question_tokens)
            context_batch = vectors.query(cont)
            question_batch = vectors.query(quest)
            pad1 = np.zeros(shape=(1, 250 - len(cont[0]), 400))
            context_batch = np.concatenate((context_batch, pad1), 1)

            pad2 = np.zeros(shape=(1, 40 - len(quest[0]), 400))
            question_batch = np.concatenate((question_batch, pad2), 1)
            answer_span = span.split()
            output1 = np.zeros(shape=(1, 250), dtype=float)
            output2 = np.zeros(shape=(1, 250), dtype=float)
            output1[0][int(answer_span[0])] = 1
            output2[0][int(answer_span[1])] = 1
            context_data.append(context_batch)
            question_data.append(question_batch)
            output_data1.append(output1)
            output_data2.append(output2)
            i += 1
            if i == data_batch:
                context_data = np.array(context_data)
                context_data = np.reshape(context_data, (data_batch, 250, 400))
                question_data = np.array(question_data)
                question_data = np.reshape(question_data, (data_batch, 40, 400))
                output_data1 = np.array(output_data1)
                output_data1 = np.reshape(output_data1, (data_batch, 250))
                output_data2 = np.array(output_data2)
                output_data2 = np.reshape(output_data2, (data_batch, 250))
                j += 1
                np.save(dir+f'numpy_data//context_data250_{j}.npy', context_data)
                np.save(dir+f'numpy_data//question_data250_{j}.npy', question_data)
                np.save(dir+f'numpy_data//output_data1_250_{j}.npy', output_data1)
                np.save(dir+f'numpy_data//output_data2_250_{j}.npy', output_data2)
                context_data = []
                question_data = []
                output_data1 = []
                output_data2 = []
                print(i*j)
                i = 0