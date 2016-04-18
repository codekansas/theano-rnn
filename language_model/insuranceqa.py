from __future__ import print_function

import os
import random
import pickle

import numpy as np
from keras.preprocessing.sequence import pad_sequences

random.seed(42)

data_path = '/media/moloch/HHD/MachineLearning/data/insuranceQA'

emb_d = pickle.load(open('word2vec.dict', 'rb'))
rev_d = dict([(v, k) for k, v in emb_d.items()])

with open(os.path.join(data_path, 'vocabulary'), 'r') as f:
    lines = f.read()

idx_d = dict()

for line in lines.split('\n'):
    if len(line) == 0: continue
    q, a = line.split('\t')
    idx_d[q] = a


def to_idx(x):
    return int(x[4:])


def convert_from_idxs(x):
    return np.asarray([emb_d[idx_d[i]] for i in x.strip().split(' ')])


def revert(x):
    return ' '.join([rev_d.get(i, 'X') for i in x])

with open(os.path.join(data_path, 'answers.label.token_idx'), 'r') as f:
    lines = f.read()

answers = dict()

for answer in lines.split('\n'):
    if len(answer) == 0: continue
    id, txt = answer.split('\t')
    id = int(id)
    answers[id] = convert_from_idxs(txt)


def get_data(f_name):
    with open(os.path.join(data_path, f_name), 'r') as f:
        lines = f.read()

    q_data = list()
    ag_data = list()
    ab_data = list()
    targets = list()

    labels = list()

    for qa_pair in lines.split('\n'):
        if len(qa_pair) == 0: continue

        if f_name == 'question.train.token_idx.label':
            q, a = qa_pair.split('\t')
        else:
            a, q, g = qa_pair.split('\t')

        good_answers = set([int(i) for i in a.strip().split(' ')])
        bad_answers = random.sample([int(i) for i in answers.keys() if i not in good_answers], len(good_answers))

        ag_data += [answers[int(i)] for i in good_answers]
        ab_data += [answers[int(i)] for i in bad_answers]

        labels += [int(i) for i in good_answers] * 2

        question = convert_from_idxs(q)
        q_data += [question] * len(good_answers)
        targets += [0] * len(bad_answers)

    # shuffle the data (i'm not sure if keras does this, but it could help generalize)
    combined = zip(q_data, ag_data, ab_data, targets, labels)
    random.shuffle(combined)
    q_data[:], ag_data[:], ab_data, targets[:], labels[:] = zip(*combined)

    q_data = pad_sequences(q_data, maxlen=maxlen, padding='post', truncating='post', value=22295)
    ag_data = pad_sequences(ag_data, maxlen=maxlen, padding='post', truncating='post', value=22295)
    ab_data = pad_sequences(ab_data, maxlen=maxlen, padding='post', truncating='post', value=22295)
    targets = np.asarray(targets)

    return q_data, ag_data, ab_data, targets

# model parameters
n_words = 22354
maxlen = 40

# the model being used
print('Generating model')

from keras_attention_model import make_model
model = make_model(maxlen, n_words, n_embed_dims=128, n_lstm_dims=256)

print('Getting data')
q_data, ag_data, ab_data, targets = get_data('question.train.token_idx.label')

print('----- Some Data -----')
print(revert(q_data[0]))
print(revert(ag_data[0]))
print(revert(ab_data[0]))

'''
Notes:
- Using the head/tail as the attention vector gave validation accuracy around 59
- The maxpooling result appears to be much better (without convolutional layer)
'''


print('Fitting model')
# training_model.load_weights('trained_iqa_model.h5')

# found through experimentation that ~24 epochs generalized the best
model.fit([q_data, ag_data, ab_data], targets, nb_epoch=24, batch_size=128, validation_split=0.2)
model.save_weights('trained_iqa_model.h5', overwrite=True)
