# theano-rnn

[Language modeling](https://github.com/codekansas/keras-language-modeling) stuff has been moved to a new repo (it didn't really match with this one anymore, and I wanted to get my commit rate up).

Some RNN implementations in Theano, along with various optimizers. In particular:

 - `vanilla.py`: Basic RNN implementation
 - `gru.py`: Gated recurrent unit
 - `lstm.py`: Long short term memory
 - `optimizers.py`: `rmsprop`, `sgd`, and `momentum` (including Nesterov momentum)

Each file has a framework for generating random data and fitting the RNN model to that data, so you can play around with it.
