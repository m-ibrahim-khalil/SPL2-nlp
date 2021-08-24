import nltk
import layers.vectors as vector
import numpy as np


class Preprocess:
    def __init__(self, context, question):
        self.context = context
        self.question = question
        self.vectors = vector.Vectors().load_vectors()

    def tokenize(self, sequence):
        tokens = [token.replace("``", '"').replace("''", '"').lower() for token in nltk.word_tokenize(sequence)]
        return tokens

    def preprocess(self, context):
        context = context.replace("''", '" ')
        context = context.replace("``", '" ')
        tokenized_context = self.tokenize(context)
        return tokenized_context

    def processForModel(self):
        context = [self.tokenize(self.context)]
        question = [self.tokenize(self.question)]
        context_len = len(context[0])
        question_len = len(question[0])
        if context_len > 250:
            context[0] = context[0][:250]
        if question_len > 20:
            question[0] = question[0][:20]
        context_batch = self.vectors.query(context)
        question_batch = self.vectors.query(question)
        # print(question_batch)
        pad1 = np.zeros(shape=(1, 250 - len(context[0]), 400))
        context_batch = np.concatenate((context_batch, pad1), 1)
        pad2 = np.zeros(shape=(1, 20 - len(question[0]), 400))
        question_batch = np.concatenate((question_batch, pad2), 1)

        return context_batch, question_batch

