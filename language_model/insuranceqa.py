from __future__ import print_function

import os
import random

import numpy as np
from keras.preprocessing.sequence import pad_sequences

random.seed(42)

data_path = '/media/moloch/HHD/MachineLearning/data/insuranceQA'

with open(os.path.join(data_path, 'vocabulary'), 'r') as f:
    lines = f.read()

# generate dictionaries
word2idx = dict()
idx2word = dict()

def to_idx(x):
    return int(x[4:])

for vocab in lines.split('\n'):
    if len(vocab) == 0: continue
    s = vocab.split('\t')
    word2idx[s[1]] = to_idx(s[0])
    idx2word[to_idx(s[0])] = s[1]

with open(os.path.join(data_path, 'answers.label.token_idx'), 'r') as f:
    lines = f.read()

answers = dict()

for answer in lines.split('\n'):
    if len(answer) == 0: continue
    id, txt = answer.split('\t')
    id = int(id)
    answers[id] = np.asarray([to_idx(i) for i in txt.split(' ')])

with open(os.path.join(data_path, 'question.train.token_idx.label'), 'r') as f:
    lines = f.read()

q_data = list()
a_data = list()
targets = list()

for qa_pair in lines.split('\n'):
    if len(qa_pair) == 0: continue
    q, a = qa_pair.split('\t')
    question = np.asarray([to_idx(i) for i in q.split(' ')])

    for answer in a.split(' '):
        # append a "bad" answer
        targets.append(0)
        q_data.append(question)
        a_data.append(random.choice(answers.values()))

        # append correct answer
        targets.append(1)
        q_data.append(question)
        a_data.append(answers[int(answer)])

# model parameters
n_words = len(word2idx) + 2
maxlen = 200

q_data = pad_sequences(q_data, maxlen)
a_data = pad_sequences(a_data, maxlen)

'''
Notes:
- Using the head/tail as the attention vector gave validation accuracy around 59
- The maxpooling result appears to be much better (without convolutional layer)
'''

# the model being used
from keras_attention_cnn_model import make_model
training_model, evaluation_model = make_model(maxlen, n_words)

print('Fitting model')
training_model.fit([q_data, a_data], targets, nb_epoch=20, batch_size=32, validation_split=0.1)
training_model.save_weights('trained_iqa_model.h5')

# TODO write evaluation component (see question.test1, question.test2)
# use evaluation model (it shares weights with the training model)
