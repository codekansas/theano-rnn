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

def get_data(f_name, set='train', small=False):
    with open(os.path.join(data_path, f_name), 'r') as f:
        lines = f.read()

    q_data = list()
    ag_data = list()
    ab_data = list()

    for qa_pair in lines.split('\n'):
        if len(qa_pair) == 0: continue

        if set == 'train':
            q, a = qa_pair.split('\t')
        elif set == 'test':
            g, q, a = qa_pair.split('\t')
        else:
            raise ValueError('`set` argument should be in (train|test)')

        question = np.asarray([to_idx(i) for i in q.split(' ')])

        for answer in a.split(' '):
            q_data.append(question)
            ab_data.append(random.choice(answers.values()))
            ag_data.append(answers[int(answer)])

            if small:
                break

    # shuffle the data (i'm not sure if keras does this, but it could help generalize)
    combined = zip(q_data, ab_data, ag_data)
    random.shuffle(combined)
    q_data[:], ab_data[:], ag_data[:] = zip(*combined)

    targets = np.asarray([0] * len(q_data))
    q_data = pad_sequences(q_data, maxlen=maxlen, padding='post')
    ab_data = pad_sequences(ab_data, maxlen=maxlen, padding='post')
    ag_data = pad_sequences(ag_data, maxlen=maxlen, padding='post')

    return q_data, ab_data, ag_data, targets


def convert(text):
    return [word2idx.get(i, len(word2idx)) for i in text.split(' ')]


def revert(ids):
    return ' '.join([idx2word.get(i, 'UNKNOWN') for i in ids])

# model parameters
n_words = len(word2idx) + 2
maxlen = 100

q_data, ab_data, ag_data, targets = get_data('question.train.token_idx.label')

print(revert(q_data[15]))
print(revert(ab_data[15]))
print(revert(ag_data[15]))

'''
Notes:
- Using the head/tail as the attention vector gave validation accuracy around 59
- The maxpooling result appears to be much better (without convolutional layer)
'''

# the model being used
from keras_attention_model import make_model
training_model, evaluation_model = make_model(maxlen, n_words, n_embed_dims=128, n_lstm_dims=256)

def evaluate():
    print('----- Evaluating on Test 1 -----')
    q_data, ab_data, ag_data, targets = get_data('question.test1.label.token_idx.pool', set='test', small=True)
    c_total = (evaluation_model.predict([q_data, ag_data]) < 0.5).sum()
    i_total = (evaluation_model.predict([q_data, ab_data]) > 0.5).sum()
    c_total /= float(len(q_data))
    i_total /= float(len(q_data))
    print('Correct answer classification rate: %f\tIncorrect answer classification rate: %f' % (c_total, i_total))

    print('----- Evaluating on Test 2 -----')
    q_data, ab_data, ag_data, targets = get_data('question.test2.label.token_idx.pool', set='test', small=True)
    c_total = (evaluation_model.predict([q_data, ag_data]) < 0.5).sum()
    i_total = (evaluation_model.predict([q_data, ab_data]) > 0.5).sum()
    c_total /= float(len(q_data))
    i_total /= float(len(q_data))
    print('Correct answer classification rate: %f\tIncorrect answer classification rate: %f' % (c_total, i_total))

    print('----- Evaluating on Dev -----')
    q_data, ab_data, ag_data, targets = get_data('question.dev.label.token_idx.pool', set='test', small=True)
    c_total = (evaluation_model.predict([q_data, ag_data]) < 0.5).sum()
    i_total = (evaluation_model.predict([q_data, ab_data]) > 0.5).sum()
    c_total /= float(len(q_data))
    i_total /= float(len(q_data))
    print('Correct answer classification rate: %f\tIncorrect answer classification rate: %f' % (c_total, i_total))

print('Fitting model')
training_model.load_weights('trained_iqa_model.h5')

for i in range(10):
    print('Iteration %d' % i)
    random.shuffle(ab_data)
    evaluate()
    training_model.fit([q_data, ag_data, ab_data], targets, nb_epoch=1, batch_size=32, validation_split=0.2)
    training_model.save_weights('trained_iqa_model.h5', overwrite=True)
