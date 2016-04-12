from __future__ import print_function

# import six.modes.cPickle as pickle
import pickle
import sys


class Dictionary:
    def __init__(self):
        self._token_counts = dict()
        self._id = 0

        self.token2id = dict()
        self.id2token = list()

    def add(self, text):
        if text is None: return

        from gensim.utils import tokenize

        if isinstance(text, str):
            docs = [tokenize(text, to_lower=True)]
        else:
            try:
                docs = [tokenize(t, to_lower=True) for t in text]
            except TypeError:
                print('Object {} is not iterable'.format(type(text)))
                sys.exit(1)

        for doc in docs:
            for t in doc:
                if t in self._token_counts:
                    self._token_counts[t] += 1
                else:
                    self._token_counts[t] = 1
                    self.id2token.append(t)
                    self.token2id[t] = self._id
                    self._id += 1

    def __getitem__(self, item):
        return self.token2id.get(item, self._id)

    def __len__(self):
        return self._id + 1

    def convert(self, text, max_len):
        from gensim.utils import tokenize
        from keras.preprocessing.sequence import pad_sequences
        from numpy import asarray

        if isinstance(text, str):
            docs = [tokenize(text, to_lower=True)]
        else:
            try:
                docs = [tokenize(t, to_lower=True) for t in text]
            except TypeError:
                print('Object {} is not iterable'.format(type(text)))
                sys.exit(1)

        return pad_sequences([asarray([self[t] for t in doc]) for doc in docs])

    def strip(self, n):
        self._token_counts = dict((k, v) for k, v in self._token_counts.items() if v > n)
        self.id2token = [k for k in self._token_counts.keys()]
        self.token2id = dict((v, k) for k, v in enumerate(self.id2token))
        self._id = len(self.id2token)

    def save(self, file_name):
        pickle.dump(self, open(file_name, 'wb'))

    @staticmethod
    def load(file_name):
        return pickle.load(open(file_name, 'rb'))

if __name__ == '__main__':
    d = Dictionary()
    d.add('the apples and oranges are very fresh today')
    print(d.id2token)
