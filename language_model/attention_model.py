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


def generate_rnn(n_in, n_out, q_emb_size, n_hidden=50, input_var=None):

    # (time_dims, input_dims)
    if input_var is None:
        X = T.matrix(name='X', dtype=dtype)
    else:
        X = input_var

    q = T.col(name='q', dtype=dtype)

    # (time_dims, output_dims)
    y = T.matrix(name='y', dtype=dtype)

    params = list()

    # input gate
    w_in_input = _get_weights('U_i', n_in, n_hidden)
    w_hidden_input = _get_weights('W_i', n_hidden, n_hidden)
    b_input = _get_zeros('b_i', n_hidden)
    params += [w_in_input, w_hidden_input, b_input]

    # forget gate
    w_in_forget = _get_weights('U_f', n_in, n_hidden)
    w_hidden_forget = _get_weights('W_f', n_hidden, n_hidden)
    b_forget = _get_zeros('b_h', n_hidden)
    params += [w_in_forget, w_hidden_forget, b_forget]

    # output gate
    w_in_output = _get_weights('U_o', n_in, n_hidden)
    w_hidden_output = _get_weights('W_o', n_hidden, n_hidden)
    b_output = _get_zeros('b_o', n_hidden)
    params += [w_in_output, w_hidden_output, b_output]

    # hidden state
    w_in_hidden = _get_weights('U_h', n_in, n_hidden)
    w_hidden_hidden = _get_weights('W_h', n_hidden, n_hidden)
    b_hidden = _get_zeros('b_o', n_hidden)
    params += [w_in_hidden, w_hidden_hidden, b_hidden]

    # output
    w_out = _get_weights('W_o', n_hidden, n_out)
    b_out = _get_zeros('b_o', n_out)
    params += [w_out, b_out]

    # starting hidden and memory unit state
    h_0 = _get_zeros('h_0', n_hidden)
    c_0 = _get_zeros('c_0', n_hidden)
    params += [h_0, c_0]

    # attention parameters
    w_am = _get_weights('W_am', n_hidden, n_hidden)
    w_qm = _get_weights('W_qm', n_hidden, n_hidden)
    w_ms = _get_weights('W_ms', n_hidden, 1)
    params += [w_am, w_qm, w_ms]

    def step(x_t, h_tm1, c_tm1):
        input_gate = T.nnet.sigmoid(T.dot(x_t, w_in_input) + T.dot(h_tm1, w_hidden_input) + b_input)
        forget_gate = T.nnet.sigmoid(T.dot(x_t, w_in_forget) + T.dot(h_tm1, w_hidden_forget) + b_forget)
        output_gate = T.nnet.sigmoid(T.dot(x_t, w_in_output) + T.dot(h_tm1, w_hidden_output) + b_output)

        candidate_state = T.tanh(T.dot(x_t, w_in_hidden) + T.dot(h_tm1, w_hidden_hidden) + b_hidden)
        memory_unit = c_tm1 * forget_gate + candidate_state * input_gate

        h_t = T.tanh(memory_unit) * output_gate

        # question embedding part
        m_a = T.tanh(T.dot(h_t, w_am) + T.dot(q, w_qm))
        s_a = T.exp(T.dot(m_a, w_ms)).sum() # sum part is to convert it
        h_t = h_t * s_a

        y_t = T.nnet.sigmoid(T.dot(h_t, w_out) + b_out)

        return h_t, memory_unit, y_t

    [_, _, output], _ = theano.scan(fn=step, sequences=X, outputs_info=[h_0, c_0, None], n_steps=X.shape[0])

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
