
# https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/

import numpy as np
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras import metrics

# load ascii text and covert to lowercase
filename = "wonderland.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()

# create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))

# summarize the loaded data
n_chars = len(raw_text)
n_vocab = len(chars)
print ("Total Characters: ", n_chars)
print ("Total Vocab: ", n_vocab)

# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])

n_patterns = len(dataX)
print ("Total Patterns: ", n_patterns)

# reshape X to be [samples, time steps, features]
X = np.reshape(dataX, (n_patterns, seq_length, 1))

# normalize
X = X / float(n_vocab)

# one hot encode the output variable
Y = np_utils.to_categorical(dataY)

######################

x = tf.placeholder(tf.float32, [20 * seq_length])
inputs = tf.reshape(x, [-1, seq_length])
inputs = tf.split(inputs, seq_length, 1)

cell = tf.nn.rnn_cell.LSTMCell(256)
outputs, states = tf.nn.static_rnn(cell=cell, inputs=inputs, dtype=tf.float32)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for ii in range(100):
    s = ii * 20
    e = (ii + 1) * 20
    xs = np.reshape(X[s:e], -1)
    [input, out, state] = sess.run([inputs, outputs, states], feed_dict={x: xs})
    print ('inputs', np.shape(input))
    print ('outputs', np.shape(out))
    print ('states', np.shape(state))
    assert(False)
    



