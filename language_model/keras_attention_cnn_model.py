from __future__ import print_function

import theano

from language_model.attention_lstm import AttentionLSTM
from language_model.get_data import get_data_set

dtype = theano.config.floatX

# get the data set
maxlen = 200 # words
q_train, a_train, t_train, q_test, a_test, t_test, n_words = get_data_set(maxlen)

##############
# Make model #
##############

from keras.layers import Input, LSTM, Embedding, merge, Convolution1D, MaxPooling1D, Dense, Flatten
from keras.models import Model

# input
question = Input(shape=(maxlen,), dtype='int32')
answer = Input(shape=(maxlen,), dtype='int32')

# language model
embedding = Embedding(output_dim=512, input_dim=n_words, input_length=maxlen)

# forward and backward lstms
f_lstm = LSTM(128, return_sequences=True)
b_lstm = LSTM(128, return_sequences=True, go_backwards=True)

# question part
q_emb = embedding(question)
q_fl = f_lstm(q_emb)
q_bl = b_lstm(q_emb)
q_out = merge([q_fl, q_bl], mode='concat', concat_axis=1)
q_out = Convolution1D(64, 3, activation='relu')(q_out)
q_out = MaxPooling1D()(q_out)
q_out = Flatten()(q_out)

# forward and backward attention lstms (paying attention to q_out)
f_lstm_attention = AttentionLSTM(128, q_out, return_sequences=True)
b_lstm_attention = AttentionLSTM(128, q_out, return_sequences=True, go_backwards=True)

# answer part
a_emb = embedding(answer)
a_fl = f_lstm_attention(a_emb)
a_bl = b_lstm_attention(a_emb)
a_out = merge([a_fl, a_bl], mode='concat', concat_axis=1)
a_out = Convolution1D(64, 3, activation='relu')(a_out)
a_out = MaxPooling1D()(a_out)
a_out = Flatten()(a_out)

q_out = Dense(128)(q_out)
a_out = Dense(128)(a_out)

# merge together
target = merge([q_out, a_out], mode='cos', dot_axes=1)
model = Model(input=[question, answer], output=target)

# need to choose binary crossentropy or mean squared error
print('Compiling model...')
model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])

print('Fitting model')
model.fit([q_train, a_train], t_train, nb_epoch=5, batch_size=32, validation_split=0.1)
model.save_weights('attention_lm_weights.h5')
