from __future__ import print_function

import theano
import theano.tensor as T
import numpy as np
from theano.tensor.signal import pool

dtype = theano.config.floatX

import optimizers
import gru as rnn


def generate_model(n_in, n_rnn_out, n_hidden):
    X, y, rnn_output, params = rnn.generate_rnn(n_in, n_rnn_out, n_hidden)
    # pool_output = T.max(rnn_output, axis=1, keepdims=True)
    pool_output = pool.pool_2d(rnn_output, (1, n_rnn_out), ignore_border=False, mode='max')

    return X, y, pool_output, params

if __name__ == '__main__':
    rng = np.random.RandomState(42)

    n_in = 10

    X, y, output, params = generate_model(n_in, 10, 10)
    output = output[-1, :]

    # minimize binary crossentropy
    # xent = -y * T.log(output) - (1 - y) * T.log(1 - output)
    # cost = xent.mean()

    # minimize mean squared error (don't get a nan error this way)
    cost = 0.5 * ((y - output) ** 2).sum()

    lr = T.scalar(name='lr', dtype=dtype)

    # grads = [T.grad(cost=cost, wrt=p) for p in params]
    updates = optimizers.rmsprop(cost, params, lr)

    t_sets = 10
    X_datas = [np.asarray(rng.rand(20, n_in) > 0.5, dtype=dtype) for _ in range(t_sets)]
    y_datas = [np.asarray(rng.rand(1, 1) > 0.5, dtype=dtype) for _ in range(t_sets)]

    train = theano.function([X, y, lr], [cost], updates=updates)
    test = theano.function([X], [output])

    n_train = 1000
    l = 0.1

    cost = sum([train(X_data, y_data, 0)[0] for X_data, y_data in zip(X_datas, y_datas)])
    print('Before training:', cost)

    for i in range(n_train):
        for X_data, y_data in zip(X_datas, y_datas):
            train(X_data, y_data, l)

        if (i + 1) % 10 == 0:
            cost = sum([train(X_data, y_data, 0)[0] for X_data, y_data in zip(X_datas, y_datas)])
            print('%d:' % (i + 1), cost)

        if (i + 1) % (n_train / 10) == 0:
            l *= 0.5
