
import numpy as np
# import tensorflow as tf
# from tensorflow.contrib import rnn
import random
import collections

from Layer import Layer 
from LSTM import LSTM 
from Dense import Dense
from Model import Model

# Text file containing words for training
training_file = 'belling_the_cat.txt'

def read_data(fname):
    with open(fname) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    content = [word for i in range(len(content)) for word in content[i].split()]
    content = np.array(content)
    return content

training_data = read_data(training_file)
print("Loaded training data...")

def build_dataset(words):
    count = collections.Counter(words).most_common()
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary

dictionary, reverse_dictionary = build_dataset(training_data)
vocab_size = len(dictionary)

# Parameters
learning_rate = 0.001
training_iters = 50000
display_step = 1000
n_input = 3

# number of units in RNN cell
n_hidden = 512

#########################
'''
inputs = tf.reshape(x, [-1, n_input])
inputs = tf.split(inputs, n_input, 1)

cell1 = rnn.BasicLSTMCell(n_hidden)
cell2 = rnn.BasicLSTMCell(n_hidden)

with tf.variable_scope('l1'):
    outputs1, states1 = tf.nn.static_rnn(cell=cell1, inputs=inputs, dtype=tf.float32)
    
with tf.variable_scope('l2'):
    outputs2, states2 = tf.nn.static_rnn(cell=cell2, inputs=outputs1, dtype=tf.float32)

pred = tf.matmul(outputs2[-1], weights['out']) + biases['out']

#########################

# Loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

# Model evaluation
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()
'''
#########################
l1 = LSTM(input_shape=(3, 1, 1), size=256)
l2 = LSTM(input_shape=(3, 1, 256), size=256, return_sequences=False)
l3 = Dense(input_shape=256, size=112)

layers = [l1, l2, l3]
model = Model(layers=layers)
#########################

# Launch the graph
step = 0
offset = random.randint(0,n_input+1)
end_offset = n_input + 1
acc_total = 0
loss_total = 0
correct_total = 0

while step < training_iters:
    # Generate a minibatch. Add some randomness on selection process.
    if offset > (len(training_data)-end_offset):
        offset = random.randint(0, n_input+1)

    symbols_in_keys = [ [dictionary[ str(training_data[i])]] for i in range(offset, offset+n_input) ]
    symbols_in_keys = np.reshape(np.array(symbols_in_keys), [n_input, -1, 1])

    symbols_out_onehot = np.zeros([vocab_size], dtype=float)
    symbols_out_onehot[dictionary[str(training_data[offset+n_input])]] = 1.0
    symbols_out_onehot = np.reshape(symbols_out_onehot,[1,-1])

    # _, acc, loss, onehot_pred = session.run([optimizer, accuracy, cost, pred], feed_dict={x: symbols_in_keys, y: symbols_out_onehot})
    correct_sum = model.train(X=symbols_in_keys, Y=symbols_out_onehot)
    correct_total += correct_sum
    
    step += 1
    offset += (n_input+1)
    
    if (step % 1000) == 0:
        acc = correct_total / 1000.0
        print ('%d/%d: %f' % (step, training_iters, acc))
        correct_total = 0



