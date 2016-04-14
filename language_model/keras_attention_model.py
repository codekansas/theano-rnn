from __future__ import print_function

##############
# Make model #
##############


def make_model(maxlen, n_words, n_lstm_dims=128, n_embed_dims=128):
    from keras.optimizers import RMSprop
    import keras.backend as K

    from language_model.attention_lstm import AttentionLSTM

    from keras.layers import Input, LSTM, Embedding, merge
    from keras.models import Model

    # input
    question = Input(shape=(maxlen,), dtype='int32')
    answer_bad = Input(shape=(maxlen,), dtype='int32')
    answer_good = Input(shape=(maxlen,), dtype='int32')

    # language model
    embedding = Embedding(output_dim=n_embed_dims, input_dim=n_words, input_length=maxlen)

    # forward and backward lstms
    f_lstm = LSTM(n_lstm_dims, dropout_U=0.2, dropout_W=0.2)
    b_lstm = LSTM(n_lstm_dims, go_backwards=True, dropout_U=0.2, dropout_W=0.2)

    # Note: Change concat_axis to 2 if return_sequences=True

    # question part
    q_emb = embedding(question)
    q_fl = f_lstm(q_emb)
    q_bl = b_lstm(q_emb)
    q_out = merge([q_fl, q_bl], mode='concat', concat_axis=1)

    # forward and backward attention lstms (paying attention to q_out)
    f_lstm_attention = AttentionLSTM(n_lstm_dims, q_out, dropout_U=0.2, dropout_W=0.2)
    b_lstm_attention = AttentionLSTM(n_lstm_dims, q_out, go_backwards=True, dropout_U=0.2, dropout_W=0.2)

    # answer part
    ab_emb = embedding(answer_bad)
    ab_fl = f_lstm_attention(ab_emb)
    ab_bl = b_lstm_attention(ab_emb)
    ab_out = merge([ab_fl, ab_bl], mode='concat', concat_axis=1)

    ag_emb = embedding(answer_good)
    ag_fl = f_lstm_attention(ag_emb)
    ag_bl = b_lstm_attention(ag_emb)
    ag_out = merge([ag_fl, ag_bl], mode='concat', concat_axis=1)

    # merge together
    target_bad = merge([q_out, ab_out], name='target_bad', mode='cos', dot_axes=1)
    target_good = merge([q_out, ag_out], name='target_good', mode='cos', dot_axes=1)

    # doing hinge loss like this means that a loss of 0 means both are classified correctly
    target = merge([target_good, target_bad], mode=lambda x: 1 - (x[0] - x[1]) / 2, output_shape=lambda x: x[0])

    training_model = Model(input=[question, answer_bad, answer_good], output=target)
    evaluation_model = Model(input=[question, answer_good], output=target_good)

    # need to choose binary crossentropy or mean squared error
    print('Compiling model...')

    optimizer = RMSprop(lr=0.0001)

    # this is more true to the paper: L = max{0, M - cosine(q, a+) + cosine(q, a-)}
    # below, "a" is a list of zeros and "b" is `target` above, i.e. 1 - cosine(q, a+) + cosine(q, a-)
    def loss(a, b):
        return K.maximum(a, b)
    # loss = 'mse'

    # unfortunately, the hinge loss approach means the "accura cy" metric isn't very valuable
    metrics = ['accuracy']

    training_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    evaluation_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return training_model, evaluation_model

if __name__ == '__main__':
    # get the data set
    maxlen = 100 # words

    from language_model.get_data import get_data_set, create_dictionary_from_qas

    dic = create_dictionary_from_qas()
    targets, questions, good, bad, n_dims = get_data_set(maxlen)

    ### THIS MODEL PERFORMS WELL ON THE TEST SET
    training_model, evaluation_model = make_model(maxlen, n_dims)

    training_model.load_weights('attention_cnn_lm_weights.h5')

    print('Fitting model')
    training_model.fit([questions, good, bad], targets, nb_epoch=5, batch_size=32, validation_split=0.2)
    training_model.save_weights('attention_cnn_lm_weights.h5', overwrite=True)

    print('----- Evaluating -----')
    total = (evaluation_model.predict([questions, good]) < 0.5).sum()
    print('Correctly classified %d correct answers' % total)
    total /= float(len(questions))
    print('Rate: %f' % total)
    total = (evaluation_model.predict([questions, bad]) > 0.5).sum()
    print('Correctly classified %d incorrect answers' % total)
    total /= float(len(questions))
    print('Rate: %f' % total)
