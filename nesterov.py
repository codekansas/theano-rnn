from __future__ import print_function

import theano
import theano.tensor as T
import numpy as np

import gru

rng = np.random.RandomState(42)
dtype = theano.config.floatX


def std_momentum(cost, param, learning_rate, momentum):
    return momentum * param - learning_rate * T.grad(cost=cost, wrt=param)


def nesterov(cost, wrt, learning_rate, momentum):
    return momentum * 

if __name__ == '__main__':
    n_in, n_out = 10, 1

    X, y, output, params = gru.generate_rnn(n_in, n_out, 50)
    output = output[-1, :]

    lr = T.scalar(name='lr', dtype=dtype)

    # minimize binary crossentropy
    xent = -y * T.log(output) - (1 - y) * T.log(1 - output)
    cost = xent.mean()

    grads = [T.grad(cost=cost, wrt=p) for p in params]
    updates = [(p, p - lr * g) for p, g in zip(params, grads)]

    t_sets = 10
    X_datas = [np.asarray(rng.rand(20, n_in) > 0.5, dtype=dtype) for _ in range(t_sets)]
    y_datas = [np.asarray(rng.rand(1, n_out) > 0.5, dtype=dtype) for _ in range(t_sets)]

    train = theano.function([X, y, lr], [cost], updates=updates)
    test = theano.function([X], [output])

    l = 0.1
    n_train = 1000

    cost = sum([train(X_data, y_data, 0)[0] for X_data, y_data in zip(X_datas, y_datas)])
    print('Before training:', cost)

    for i in range(n_train):
        for X_data, y_data in zip(X_datas, y_datas):
            train(X_data, y_data, l)

        if (i+1) % (n_train / 5) == 0:
            cost = sum([train(X_data, y_data, 0)[0] for X_data, y_data in zip(X_datas, y_datas)])
            print('%d (lr = %f):' % (i+1, l), cost)
            l *= 0.5
