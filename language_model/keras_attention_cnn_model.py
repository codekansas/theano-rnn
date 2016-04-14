from __future__ import print_function

import theano
from keras.engine import Merge
from keras.optimizers import RMSprop

from language_model.attention_lstm import AttentionLSTM
from language_model.get_data import get_data_set

dtype = theano.config.floatX

##############
# Make model #
##############

from keras.layers import Input, LSTM, Embedding, merge, Convolution1D, Dense, Flatten, AveragePooling1D
from keras.models import Model


def make_model(maxlen, n_words, n_lstm_dims=128, n_output_dims=128, n_embed_dims=512, n_conv_filters=64, conv_len=3):
    # input
    question = Input(shape=(maxlen,), dtype='int32')
    answer_bad = Input(shape=(maxlen,), dtype='int32')
    answer_good = Input(shape=(maxlen,), dtype='int32')

    # language model
    embedding = Embedding(output_dim=n_embed_dims, input_dim=n_words, input_length=maxlen)

    # forward and backward lstms
    f_lstm = LSTM(n_lstm_dims) #, return_sequences=True)
    b_lstm = LSTM(n_lstm_dims, go_backwards=True) #, return_sequences=True)

    # convolution / maxpooling layers
    # conv = Convolution1D(n_conv_filters, conv_len, activation='relu')
    pool = AveragePooling1D()
    flat = Flatten()

    # question part
    q_emb = embedding(question)
    q_fl = f_lstm(q_emb)
    q_bl = b_lstm(q_emb)
    q_out = merge([q_fl, q_bl], mode='concat', concat_axis=1)
    # q_out = conv(q_out)
    # q_out = pool(q_out)
    # q_out = flat(q_out)

    # forward and backward attention lstms (paying attention to q_out)
    f_lstm_attention = AttentionLSTM(n_lstm_dims, q_out) #, return_sequences=True)
    b_lstm_attention = AttentionLSTM(n_lstm_dims, q_out, go_backwards=True) #, return_sequences=True)

    conv_to_out = Dense(n_output_dims)

    # answer part
    ab_emb = embedding(answer_bad)
    ab_fl = f_lstm_attention(ab_emb)
    ab_bl = b_lstm_attention(ab_emb)
    ab_out = merge([ab_fl, ab_bl], mode='concat', concat_axis=1)
    # a_out = conv(a_out)
    # ab_out = pool(ab_out)
    # ab_out = flat(ab_out)
    # ab_out = conv_to_out(ab_out)

    ag_emb = embedding(answer_good)
    ag_fl = f_lstm_attention(ag_emb)
    ag_bl = b_lstm_attention(ag_emb)
    ag_out = merge([ag_fl, ag_bl], mode='concat', concat_axis=1)
    # a_out = conv(a_out)
    # ag_out = pool(ag_out)
    # ag_out = flat(ag_out)
    # ag_out = conv_to_out(ag_out)

    # q_out = Dense(n_output_dims)(q_out)

    # merge together
    target_bad = merge([q_out, ab_out], name='target_bad', mode='cos', dot_axes=1)
    target_good = merge([q_out, ag_out], name='target_good', mode='cos', dot_axes=1)

    # doing hinge loss like this means that a loss of 0 means both are classified correctly
    target = merge([target_good, target_bad], mode=lambda x: 1 - (x[0] - x[1]) / 2, output_shape=lambda x: x[0])

    training_model = Model(input=[question, answer_bad, answer_good], output=target)
    evaluation_model = Model(input=[question, answer_good], output=target_good)

    # need to choose binary crossentropy or mean squared error
    print('Compiling model...')
    optimizer = RMSprop(lr=0.001)
    loss = 'mse'
    metrics = ['accuracy']
    training_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    evaluation_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return training_model, evaluation_model

if __name__ == '__main__':
    # get the data set
    maxlen = 200 # words
    targets, questions, good, bad, n_words = get_data_set(maxlen)

    training_model, evaluation_model = make_model(maxlen, n_words)

    print('Fitting model')
    training_model.fit([questions, good, bad], targets, nb_epoch=5, batch_size=32, validation_split=0.1)
    training_model.save_weights('attention_cnn_lm_weights.h5')
