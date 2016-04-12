from __future__ import print_function

import theano
import theano.tensor as T

dtype = theano.config.floatX


def rmsprop(cost, params, learning_rate, rho=0.9, epsilon=1e-6):
    updates = list()

    for param in params:
        accu = theano.shared(np.zeros(param.get_value(borrow=True).shape, dtype=dtype),
                             broadcastable=param.broadcastable)

        grad = T.grad(cost, param)
        accu_new = rho * accu + (1 - rho) * grad ** 2

        updates.append((accu, accu_new))
        updates.append((param, param - (learning_rate * grad / T.sqrt(accu_new + epsilon))))

    return updates

def sgd(cost, params, learning_rate):
    return [(param, param - learning_rate * T.grad(cost, param)) for param in params]


def momentum(cost, params, learning_rate, momentum=0.9, type='nesterov'):
    assert type in ['std', 'nesterov'], 'Possible momentum types: `std`, `nesterov`'
    assert 0 <= momentum < 1
    updates = list()

    for param in params:
        # this is the "momentum" part: it is shared across updates
        velocity = theano.shared(np.zeros(param.get_value(borrow=True).shape, dtype=dtype),
                                 broadcastable=param.broadcastable)

        update = param - learning_rate * T.grad(cost, param)
        if type == 'nesterov': # nesterov
            x = momentum * velocity + update - param
            updates.append((velocity, x))
            updates.append((param, momentum * x + update))
        else: # standard
            x = momentum * velocity + update
            updates.append((velocity, x - param))
            updates.append((param, x))

    return updates

if __name__ == '__main__':
    import lstm as rnn
    import numpy as np

    rng = np.random.RandomState(42)

    n_in, n_out = 10, 1

    X, y, output, params = rnn.generate_rnn(n_in, n_out, 50)
    output = output

    # minimize binary crossentropy
    xent = -y * T.log(output) - (1 - y) * T.log(1 - output)
    cost = xent.mean()

    lr = T.scalar(name='lr', dtype=dtype)

    # grads = [T.grad(cost=cost, wrt=p) for p in params]
    updates = rmsprop(cost, params, lr)

    t_sets = 10
    X_datas = [np.asarray(rng.rand(20, n_in) > 0.5, dtype=dtype) for _ in range(t_sets)]
    y_datas = [np.asarray(rng.rand(20, n_out) > 0.5, dtype=dtype) for _ in range(t_sets)]

    train = theano.function([X, y, lr], [cost], updates=updates)
    test = theano.function([X], [output])

    n_train = 1000
    l = 0.1

    cost = sum([train(X_data, y_data, 0)[0] for X_data, y_data in zip(X_datas, y_datas)])
    print('Before training:', cost)

    for i in range(n_train):
        for X_data, y_data in zip(X_datas, y_datas):
            train(X_data, y_data, l)

        if (i+1) % 10 == 0:
            cost = sum([train(X_data, y_data, 0)[0] for X_data, y_data in zip(X_datas, y_datas)])
            print('%d:' % (i+1), cost)
