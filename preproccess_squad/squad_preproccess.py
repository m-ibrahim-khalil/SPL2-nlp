"""Downloads SQuAD train and dev sets, preprocessed and writes tokenized versions to file"""

import os
import random
import json
import nltk
import numpy as np
from tqdm import tqdm

random.seed(42)
np.random.seed(42)


def write_to_file(out_file, line):
    line += '\n'
    encoded_line = str.encode(line)
    out_file.write(encoded_line)


def data_from_json(filename):
    with open(filename, encoding='utf8') as data_file:
        data = json.load(data_file)
    return data


def tokenize(sequence):
    tokens = [token.replace("``", '"').replace("''", '"').lower() for token in nltk.word_tokenize(sequence)]
    return tokens


def total_exs(dataset):
    total = 0
    for article in dataset['data']:
        for para in article['paragraphs']:
            total += len(para['qas'])
    return total


def reporthook(t):
    """https://github.com/tqdm/tqdm"""
    last_b = [0]

    def inner(b=1, bsize=1, t_size=None):
        if t_size is not None:
            t.total = t_size
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return inner


def get_char_word_loc_mapping(context, context_tokens):
    accumulator = ''
    current_token_idx = 0  # current word loc
    mapping = dict()

    for char_idx, char in enumerate(context):  # step through original characters
        if char != u' ' and char != u'\n':  # if it's not a space:
            accumulator += char  # add to accumulator
            context_token = (context_tokens[current_token_idx])  # current word token
            if accumulator == context_token:  # if the accumulator now matches the current word token
                syn_start = char_idx - len(accumulator) + 1  # char loc of the start of this word
                for char_loc in range(syn_start, char_idx + 1):
                    mapping[char_loc] = (accumulator, current_token_idx)  # add to mapping
                accumulator = ''  # reset accumulator
                current_token_idx += 1

    if current_token_idx != len(context_tokens):
        return None
    else:
        return mapping


def preprocess_and_write(dataset, tier, out_dir):
    num_examples = 0
    num_mapping_prob, num_token_prob, num_span_align_prob = 0, 0, 0
    examples = []

    for articles_id in tqdm(range(len(dataset['data'])), desc="Preprocessing {}".format(tier)):

        article_paragraphs = dataset['data'][articles_id]['paragraphs']
        for pid in range(len(article_paragraphs)):

            context = article_paragraphs[pid]['context']  # string

            # The following replacements are suggested in the paper
            # BidAF (Seo et al., 2016)
            context = context.replace("''", '" ')
            context = context.replace("``", '" ')

            context_tokens = tokenize(context)  # list of strings (lowercase)
            context = context.lower()

            qas = article_paragraphs[pid]['qas']  # list of questions

            charloc2wordloc = get_char_word_loc_mapping(context, context_tokens)  # charloc2wordloc maps the character
            # location (int) of a context token to a pair giving (word (string), word loc (int)) of that token

            if charloc2wordloc is None:  # there was a problem
                num_mapping_prob += len(qas)
                continue  # skip this context example

            # for each question, process the question and answer and write to file
            for qn in qas:

                # read the question text and tokenize
                question = (qn['question'])  # string
                question_tokens = tokenize(question)  # list of strings

                # of the three answers, just take the first
                ans_text = (qn['answers'][0]['text']).lower()  # get the answer text
                ans_start_char_loc = qn['answers'][0]['answer_start']  # answer start loc (character count)
                ans_end_char_loc = ans_start_char_loc + len(ans_text)  # answer end loc (character count) (exclusive)

                # Check that the provided character spans match the provided answer text
                if context[ans_start_char_loc:ans_end_char_loc] != ans_text:
                    num_span_align_prob += 1
                    continue

                # get word locations for answer start and end (inclusive)
                ans_start_word_loc = charloc2wordloc[ans_start_char_loc][1]  # answer start word loc
                ans_end_word_loc = charloc2wordloc[ans_end_char_loc - 1][1]  # answer end word loc
                assert ans_start_word_loc <= ans_end_word_loc
                ans_tokens = context_tokens[ans_start_word_loc:ans_end_word_loc + 1]
                if "".join(ans_tokens) != "".join(ans_text.split()):
                    num_token_prob += 1
                    continue

                examples.append((' '.join(context_tokens), ' '.join(question_tokens), ' '.join(ans_tokens),
                                 ' '.join([str(ans_start_word_loc), str(ans_end_word_loc)])))

                num_examples += 1

    print("Number of (context, question, answer) triples discarded due to char -> token mapping problems: ",
          num_mapping_prob)
    print("Number of (context, question, answer) triples discarded because character-based answer span is unaligned "
          "with tokenization: ", num_token_prob)
    print("Number of (context, question, answer) triples discarded due character span alignment problems (usually "
          "Unicode problems): ", num_span_align_prob)
    print("Processed %i examples of total %i\n" % (
        num_examples, num_examples + num_mapping_prob + num_token_prob + num_span_align_prob))

    indices = list(range(len(examples)))
    np.random.shuffle(indices)

    with open(os.path.join(out_dir, tier + '.context'), 'wb') as context_file, \
            open(os.path.join(out_dir, tier + '.question'), 'wb') as question_file, \
            open(os.path.join(out_dir, tier + '.answer'), 'wb') as ans_text_file, \
            open(os.path.join(out_dir, tier + '.span'), 'wb') as span_file:

        for i in indices:
            (context, question, answer, answer_span) = examples[i]
            write_to_file(context_file, context)
            write_to_file(question_file, question)
            write_to_file(ans_text_file, answer)
            write_to_file(span_file, answer_span)


def main():
    train_filename = "train-v1.1.json"
    dev_filename = "dev-v1.1.json"
    out_dir = "F://Pycharm Projects//Spl2-nlp-QA//preproccess_squad"

    train_data = data_from_json(train_filename)
    print("Train data has %i examples total" % total_exs(train_data))
    preprocess_and_write(train_data, 'train', out_dir)

    dev_data = data_from_json(dev_filename)
    print("Dev data has %i examples total" % total_exs(dev_data))
    preprocess_and_write(dev_data, 'dev', out_dir)


if __name__ == '__main__':
    main()
