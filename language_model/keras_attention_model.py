from __future__ import print_function

import theano
from keras.optimizers import RMSprop

from language_model.attention_lstm import AttentionLSTM
from language_model.get_data import get_data_set

dtype = theano.config.floatX

##############
# Make model #
##############

from keras.layers import Input, LSTM, Embedding, merge
from keras.models import Model


def make_model(maxlen, n_words, n_lstm_dims=128, n_alstm_dims=128, n_embed_dims=512):
    # input
    question = Input(shape=(maxlen,), dtype='int32')
    answer = Input(shape=(maxlen,), dtype='int32')

    # language model
    embedding = Embedding(output_dim=n_embed_dims, input_dim=n_words, input_length=maxlen)

    f_lstm = LSTM(n_lstm_dims)
    b_lstm = LSTM(n_lstm_dims, go_backwards=True)

    # question part
    q_emb = embedding(question)
    q_fl = f_lstm(q_emb)
    q_bl = b_lstm(q_emb)
    q_out = merge([q_fl, q_bl], mode='concat', concat_axis=1)

    f_lstm_attention = AttentionLSTM(n_alstm_dims, q_out)
    b_lstm_attention = AttentionLSTM(n_alstm_dims, q_out, go_backwards=True)

    # answer part
    a_emb = embedding(answer)
    a_fl = f_lstm_attention(a_emb)
    a_bl = b_lstm_attention(a_emb)
    a_out = merge([a_fl, a_bl], mode='concat', concat_axis=1)

    # merge together
    target = merge([q_out, a_out], mode='cos', dot_axes=1)
    model = Model(input=[question, answer], output=target)

    # need to choose binary crossentropy or mean squared error
    rmsprop = RMSprop(lr=0.0001)
    model.compile(optimizer=rmsprop, loss='binary_crossentropy', metrics=['accuracy'])

    return model

if __name__ == '__main__':
    # get the data set
    maxlen = 200 # words
    q_train, a_train, t_train, q_test, a_test, t_test, n_words = get_data_set(maxlen)

    model = make_model(maxlen, n_words)

    print('Fitting model')
    model.fit([q_train, a_train], t_train, nb_epoch=5, batch_size=32, validation_split=0.1)
    model.save_weights('attention_lm_weights.h5')
