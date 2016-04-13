from __future__ import print_function

import theano
from keras.optimizers import RMSprop

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

# parameters
n_lstm_dims = 128
n_output_dims = 128

# input
question = Input(shape=(maxlen,), dtype='int32')
answer = Input(shape=(maxlen,), dtype='int32')

# language model
embedding = Embedding(output_dim=512, input_dim=n_words, input_length=maxlen)

# forward and backward lstms
f_lstm = LSTM(n_lstm_dims, return_sequences=True)
b_lstm = LSTM(n_lstm_dims, return_sequences=True, go_backwards=True)

# convolution / maxpooling layers
conv = Convolution1D(64, 3, activation='relu')
maxpool = MaxPooling1D()
flat = Flatten()

# question part
q_emb = embedding(question)
q_fl = f_lstm(q_emb)
q_bl = b_lstm(q_emb)
q_out = merge([q_fl, q_bl], mode='concat', concat_axis=1)
q_out = conv(q_out)
q_out = maxpool(q_out)
q_out = flat(q_out)

# forward and backward attention lstms (paying attention to q_out)
f_lstm_attention = AttentionLSTM(n_lstm_dims, q_out, return_sequences=True)
b_lstm_attention = AttentionLSTM(n_lstm_dims, q_out, return_sequences=True, go_backwards=True)

# answer part
a_emb = embedding(answer)
a_fl = f_lstm_attention(a_emb)
a_bl = b_lstm_attention(a_emb)
a_out = merge([a_fl, a_bl], mode='concat', concat_axis=1)
a_out = conv(a_out)
a_out = maxpool(a_out)
a_out = flat(a_out)

conv_to_out = Dense(n_output_dims)
q_out = conv_to_out(q_out)
a_out = conv_to_out(a_out)

# merge together
target = merge([q_out, a_out], mode='cos', dot_axes=1)
model = Model(input=[question, answer], output=target)

# need to choose binary crossentropy or mean squared error
print('Compiling model...')
rmsprop = RMSprop(lr=0.0001)
model.compile(optimizer=rmsprop, loss='binary_crossentropy', metrics=['accuracy'])

print('Fitting model')
model.fit([q_train, a_train], t_train, nb_epoch=5, batch_size=32, validation_split=0.1)
model.save_weights('attention_cnn_lm_weights.h5')
