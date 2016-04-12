from __future__ import print_function

import re
import os

import theano
import theano.tensor as T
import numpy as np

import optimizers
from dictionary import Dictionary
from attention_model import AttentionLSTM

dtype = theano.config.floatX
rng = np.random.RandomState(42)

path = '/home/moloch/Documents/PycharmProjects/ml/theano_stuff/ir/LiveQA2015-qrels-ver2.txt'

#################
# Load QA pairs #
#################

with open(path, 'r') as f:
    lines = f.read().split('\r')
print('%d graded answers' % len(lines))

questions = dict()
answers = dict()

qpattern = re.compile('(\d+)q\t([\w\d]+)\t\t([^\t]+)\t(.*?)\t([^\t]+)\t([^\t]+)$')
for line in lines:
    qm = qpattern.match(line)
    if qm:
        trecid = qm.group(1)
        qid = qm.group(2)
        title = qm.group(3)
        content = qm.group(4)
        maincat = qm.group(5)
        subcat = qm.group(6)
        questions[trecid] = { 'qid': qid, 'title': title, 'content': content, 'maincat': maincat, 'subcat': subcat }
        answers[trecid] = list()
    else:
        trecid, qid, score, answer, resource = line.split('`\t`')
        trecid = trecid[:-1]
        answers[trecid].append({ 'score': score, 'answer': answer, 'resource': resource })

assert len(questions) == len(answers) == 1087, 'There was an error processing the file somewhere (should have 1087 questions)'

#####################
# Create dictionary #
#####################

dict_path = '/home/moloch/Documents/PycharmProjects/ml/theano_stuff/ir/dict.pkl'
if os.path.exists(dict_path):
    dic = Dictionary.load(dict_path)
else:
    dic = Dictionary()
    for q in questions.values():
        dic.add(q['content'])
        dic.add(q['title'])

    for aa in answers.values():
        for a in aa:
            dic.add(a['answer'])
    dic.save(dict_path)

##############
# Make model #
##############

from keras.layers import Input, LSTM, Embedding, merge
from keras.models import Model

# input
maxlen = 200 # words
question = Input(shape=(maxlen,), dtype='int32')
answer = Input(shape=(maxlen,), dtype='int32')

# language model
embedding = Embedding(output_dim=512, input_dim=len(dic), input_length=maxlen)

f_lstm = LSTM(128)
b_lstm = LSTM(128, go_backwards=True)

# question part
q_emb = embedding(question)
q_fl = f_lstm(q_emb)
q_bl = b_lstm(q_emb)
q_out = merge([q_fl, q_bl], mode='concat', concat_axis=1)

f_lstm_attention = AttentionLSTM(128, q_out)
b_lstm_attention = AttentionLSTM(128, q_out, go_backwards=True)

# good answer part
a_emb = embedding(answer)
a_fl = f_lstm_attention(a_emb)
a_bl = b_lstm_attention(a_emb)
a_out = merge([a_fl, a_bl], mode='concat', concat_axis=1)

# merge together
target = merge([q_out, a_out], mode='cos', dot_axes=1)
model = Model(input=[question, answer], output=target)

# need to choose binary crossentropy or mean squared error
print('Compiling model...')
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

################
# Train model #
###############

q_test, q_train = list(), list()
a_test, a_train = list(), list()
t_test, t_train = list(), list()

for id, question in questions.items():
    gans, bans = list(), list()
    qc = dic.convert(question['title'] + question['content'], maxlen=maxlen)[0]

    for answer in answers[id]:
        ac = dic.convert(answer['answer'], maxlen=maxlen)[0]
        score = int(answer['score'])

        if rng.rand() < 0.1:
            q_test.append(qc)
            a_test.append(ac)
            t_test.append(0 if score < 3 else 1)
        else:
            q_train.append(qc)
            a_train.append(ac)
            t_train.append(0 if score < 3 else 1)

q_test = np.asarray(q_test)
q_train = np.asarray(q_train)
a_test = np.asarray(a_test)
a_train = np.asarray(a_train)
t_test = np.asarray(t_test)
t_train = np.asarray(t_train)

print('Fitting model')
model.fit([q_train, a_train], t_train, nb_epoch=5, batch_size=32, validation_split=0.1)
model.save_weights(os.path.join('/home/moloch/Documents/PycharmProjects/theano-rnn', 'attention_lm_weights.h5'))
