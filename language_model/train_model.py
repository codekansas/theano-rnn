from __future__ import print_function

import re
import os

import theano
import theano.tensor as T
import numpy as np

import optimizers
from dictionary import Dictionary

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


def generate_model(n_in, embedding_dims, context_win_size, n_rnn_out, n_hidden):
    import gru as rnn

    ################
    # Model itself #
    ################

    embeddings = theano.shared(name='embeddings', value=0.2 * np.random.uniform(-1., 1., (n_in+1, embedding_dims)).astype(dtype))

    idxs_1 = T.ivector()
    X_1 = embeddings[idxs_1].reshape((idxs_1.shape[0], embedding_dims * context_win_size))

    idxs_2 = T.ivector()
    X_2 = embeddings[idxs_2].reshape((idxs_2.shape[0], embedding_dims * context_win_size))

    idxs_3 = T.ivector()
    X_3 = embeddings[idxs_3].reshape((idxs_3.shape[0], embedding_dims * context_win_size))

    X_1, y_1, rnn_output_1, params = rnn.generate_rnn(embedding_dims, n_rnn_out, n_hidden, input_var=X_1)
    X_2, y_2, rnn_output_2, params = rnn.generate_rnn(embedding_dims, n_rnn_out, n_hidden, input_var=X_2, share_params=params)
    X_3, y_3, rnn_output_3, params = rnn.generate_rnn(embedding_dims, n_rnn_out, n_hidden, input_var=X_3, share_params=params)

    output_1 = rnn_output_1[-1, :]
    output_2 = rnn_output_2[-1, :]
    output_3 = rnn_output_3[-1, :]

    ################
    # For training #
    ################

    lr = T.scalar(name='lr', dtype=dtype)
    margin = T.scalar(name='margin', dtype=dtype)

    def cossim(a, b):
        return T.dot(a, b.T).sum() / T.sqrt((a ** 2).sum() * (b ** 2).sum())

    # maximize cos similarity between question and ground truth
    cost = T.maximum(0, margin - cossim(output_1, output_2) + cossim(output_1, output_3))
    updates = optimizers.rmsprop(cost, params, lr)

    normalize = theano.function(inputs=[], updates={embeddings: embeddings / T.sqrt((embeddings ** 2).sum(axis=1).dimshuffle(0, 'x'))})

    # train(question, good, bad, lr, margin)
    print('Creating train / test functions')
    train = theano.function([idxs_1, idxs_2, idxs_3, lr, margin], [cost], updates=updates)
    test = theano.function([idxs_1, idxs_2], [cossim(idxs_1, idxs_2)])

    return normalize, train, test

###############
# Train model #
###############

normalize, train, test = generate_model(len(dic), 128, 5, 128, 256)

qs_train, qs_test = list(), list()

for id, question in questions.items():
    gans, bans = list(), list()
    qc = question['title'] + question['content']

    for answer in answers[id]:
        ac = answer['answer']
        score = answer['score']

        if int(score) >= 3:
            gans.append(dic.convert(ac)[0])
        else:
            bans.append(dic.convert(ac)[0])

    if rng.rand() < 0.1:
        qs_test.append([dic.convert(qc)[0], [gans, bans]])
    else:
        qs_train.append([dic.convert(qc)[0], [gans, bans]])

l = 0.1
margin = 0.5
n_epochs = 100

for epoch in range(n_epochs):
    print('Epoch %d' % epoch)

    for qn in qs_train:
        q = qn[0]
        for g, b in zip(qn[1][0], qn[1][1]):
            print(q.shape, g.shape, b.shape)
            train(q, g, b, l, margin)

    for qn in qs_test:
        cost = 0
        q = qn[0]
        for g, b in zip(qn[1][0], qn[1][1]):
            cost += train(q, g, b, l, margin)[0].sum()

    l *= 0.95
