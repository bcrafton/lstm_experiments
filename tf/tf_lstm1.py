'''
A Recurrent Neural Network (LSTM) implementation example using TensorFlow..
Next word prediction after n_input words learned from text file.
A story is automatically generated if the predicted word is fed back as input.

Author: Rowel Atienza
Project: https://github.com/roatienza/Deep-Learning-Experiments
'''

from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import random
import collections
import time

start_time = time.time()
def elapsed(sec):
    if sec<60:
        return str(sec) + " sec"
    elif sec<(60*60):
        return str(sec/60) + " min"
    else:
        return str(sec/(60*60)) + " hr"


# Target log path
logs_path = '/tmp/tensorflow/rnn_words'
writer = tf.summary.FileWriter(logs_path)

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

# tf Graph input
x = tf.placeholder("float", [None, n_input, 1])
y = tf.placeholder("float", [None, vocab_size])

#########################

'''
def RNN(x, weights, biases):

    # reshape to [1, n_input]
    x = tf.reshape(x, [-1, n_input])

    # Generate a n_input-element sequence of inputs
    # (eg. [had] [a] [general] -> [20] [6] [33])
    x = tf.split(x,n_input,1)

    # 2-layer LSTM, each layer has n_hidden units.
    # Average Accuracy= 95.20% at 50k iter
    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden),rnn.BasicLSTMCell(n_hidden)])

    # 1-layer LSTM with n_hidden units but with lower accuracy.
    # Average Accuracy= 90.60% 50k iter
    # Uncomment line below to test but comment out the 2-layer rnn.MultiRNNCell above
    # rnn_cell = rnn.BasicLSTMCell(n_hidden)

    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

    # there are n_input outputs but
    # we only want the last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases)
'''

#########################

inputs = tf.reshape(x, [-1, n_input])
inputs = tf.split(inputs, n_input, 1)

with tf.variable_scope('l1'):
    cell1 = rnn.BasicLSTMCell(n_hidden)
    outputs1, states1 = tf.nn.static_rnn(cell=cell1, inputs=inputs, dtype=tf.float32)
    
with tf.variable_scope('l2'):
    cell2 = rnn.BasicLSTMCell(n_hidden)
    outputs2, states2 = tf.nn.static_rnn(cell=cell2, inputs=outputs1, dtype=tf.float32)

with tf.variable_scope('l3'):
    weights = {'out': tf.Variable(tf.random_normal([n_hidden, vocab_size]))}
    biases = {'out': tf.Variable(tf.random_normal([vocab_size]))}
    pred = tf.matmul(outputs2[-1], weights['out']) + biases['out']

weights_l1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='l1')
weights_l2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='l2')
weights_l3 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='l3')

#########################

# Loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

# Model evaluation
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as session:
    session.run(init)
    step = 0
    offset = random.randint(0,n_input+1)
    end_offset = n_input + 1
    acc_total = 0
    loss_total = 0

    writer.add_graph(session.graph)

    while step < training_iters:
        # Generate a minibatch. Add some randomness on selection process.
        if offset > (len(training_data)-end_offset):
            offset = random.randint(0, n_input+1)

        symbols_in_keys = [ [dictionary[ str(training_data[i])]] for i in range(offset, offset+n_input) ]
        symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])

        symbols_out_onehot = np.zeros([vocab_size], dtype=float)
        symbols_out_onehot[dictionary[str(training_data[offset+n_input])]] = 1.0
        symbols_out_onehot = np.reshape(symbols_out_onehot,[1,-1])

        _, acc, loss, onehot_pred, l1, l2, l3 = session.run([optimizer, accuracy, cost, pred, weights_l1, weights_l2, weights_l3], feed_dict={x: symbols_in_keys, y: symbols_out_onehot})
        
        for w in l1:
            if len(np.shape(w)) == 2:
                print ('l1', np.shape(w), np.std(w), np.mean(w))
        for w in l2:
            if len(np.shape(w)) == 2:
                print ('l2', np.shape(w), np.std(w), np.mean(w))
        for w in l3:
            if len(np.shape(w)) == 2:
                print ('l3', np.shape(w), np.std(w), np.mean(w))
        assert(False)
        
        loss_total += loss
        acc_total += acc
        if (step+1) % display_step == 0:
            print("Iter= " + str(step+1) + ", Average Loss= " + \
                  "{:.6f}".format(loss_total/display_step) + ", Average Accuracy= " + \
                  "{:.2f}%".format(100*acc_total/display_step))
            acc_total = 0
            loss_total = 0
            symbols_in = [training_data[i] for i in range(offset, offset + n_input)]
            symbols_out = training_data[offset + n_input]
            symbols_out_pred = reverse_dictionary[int(tf.argmax(onehot_pred, 1).eval())]
            print("%s - [%s] vs [%s]" % (symbols_in,symbols_out,symbols_out_pred))
        step += 1
        offset += (n_input+1)
        
    print("Optimization Finished!")
    print("Elapsed time: ", elapsed(time.time() - start_time))

