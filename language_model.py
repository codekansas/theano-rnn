from __future__ import print_function

import theano
import theano.tensor as T
import numpy as np
from theano.tensor.signal import pool

import optimizers
import gru as rnn


def generate_model(n_in, n_out, n_hidden):
    maxpool_shape = (1, n_out) # really just the max of each recurrent step

    X, y, rnn_output, params = rnn.generate_rnn(n_in, n_out, n_hidden)
    pool_output = pool.pool_2d(rnn_output, maxpool_shape, ignore_border=False, mode='max')

    return X, y, pool_output, params

if __name__ == '__main__':
    # TODO
    pass
