from __future__ import print_function

import theano
from theano import tensor as T
import numpy as np

rng = np.random.RandomState(42)
dtype = theano.config.floatX


def _get_weights(name, *shape, **kwargs):
    """ Initialize a weight matrix of size `n_in` by `n_out` with random values from `low` to `high` """
    low, high = kwargs.get('low', -1), kwargs.get('high', 1)
    return theano.shared(np.asarray(rng.rand(*shape) * (high - low) + low, dtype=dtype), name=name, borrow=True)


def _get_zeros(name, *shape, **kwargs):
    return theano.shared(np.zeros(shape=shape, dtype=dtype), name=name, borrow=True)


def generate_rnn(n_in, n_out, n_hidden=50):

    # (time_dims, input_dims)
    X = T.matrix(name='X', dtype=dtype)

    params = list()

    # update gate
    w_in_update = _get_weights('U_z', n_in, n_hidden)
    w_hidden_update = _get_weights('W_z', n_hidden, n_hidden)
    b_update = _get_zeros('b_z', n_hidden)
    params += [w_in_update, w_hidden_update, b_update]

    # reset gate
    w_in_reset = _get_weights('U_r', n_in, n_hidden)
    w_hidden_reset = _get_weights('W_r', n_hidden, n_hidden)
    b_reset = _get_weights('b_r', n_hidden)
    params += [w_in_reset, w_hidden_reset, b_reset]

    # hidden layer
    w_in_hidden = _get_weights('U_h', n_in, n_hidden)
    w_reset_hidden = _get_weights('W_h', n_hidden, n_hidden)
    b_in_hidden = _get_zeros('b_h', n_hidden)
    params += [w_in_hidden, w_reset_hidden, b_in_hidden]

    # output
    w_out = _get_weights('W_o', n_hidden, n_out)
    b_out = _get_zeros('b_o', n_out)
    params += [w_out, b_out]

    # starting hidden state
    h_0 = _get_zeros('h_0', n_hidden)
    params += [h_0]

    def step(x_t, h_tm1):
        update_gate = T.nnet.sigmoid(T.dot(x_t, w_in_update) + T.dot(h_tm1, w_hidden_update) + b_update)
        reset_gate = T.nnet.sigmoid(T.dot(x_t, w_in_reset) + T.dot(h_tm1, w_hidden_reset) + b_reset)
        h_t_temp = T.tanh(T.dot(x_t, w_in_hidden) + T.dot(h_tm1 * reset_gate, w_reset_hidden) + b_in_hidden)

        h_t = (1 - update_gate) * h_t_temp + update_gate * h_tm1
        y_t = T.nnet.sigmoid(T.dot(h_t, w_out) + b_out)

        return h_t, y_t

    [_, output], _ = theano.scan(fn=step, sequences=X, outputs_info=[h_0, None], n_steps=X.shape[0])

    return X, y, output, params


if __name__ == '__main__':
    import optimizers

    n_in, n_out = 10, 1

    X, y, output, params = generate_rnn(n_in, n_out, 50)
    output = output[-1, :]

    lr = T.scalar(name='lr', dtype=dtype)

    # minimize binary crossentropy
    xent = -y * T.log(output) - (1 - y) * T.log(1 - output)
    cost = xent.mean()

    updates = optimizers.rmsprop(cost, params, lr)

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
