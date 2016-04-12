from __future__ import print_function

import re
import os

import language_model

path = '/home/moloch/Documents/PycharmProjects/ml/theano_stuff/ir/LiveQA2015-qrels-ver2.txt'

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

dict_path = '/home/moloch/Documents/PycharmProjects/ml/theano_stuff/ir/dict.pkl'
if os.path.exists(dict_path):
    dict = Dictionary.load(dict_path)
else:
    dict = Dictionary()
    for q in questions.values():
        dict.add(q['content'])
        dict.add(q['title'])

print(dict)
