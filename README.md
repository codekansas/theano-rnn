# theano-rnn

Some RNN implementations in Theano, along with various optimizers. In particular:

 - `vanilla.py`: Basic RNN implementation
 - `gru.py`: Gated recurrent unit
 - `lstm.py`: Long short term memory
 - `optimizers.py`: `rmsprop`, `sgd`, and `momentum` (including Nesterov momentum)

Each file has a framework for generating random data and fitting the RNN model to that data, so you can play around with it.

I used to have some language modeling stuff here using Keras. I moved all that stuff to its own repository, which can be found [here](https://github.com/codekansas/keras-language-modeling).
