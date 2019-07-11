
import numpy as np
import tensorflow as tf
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

#########################

x = tf.placeholder("float", [n_input, None, 1])
y = tf.placeholder("float", [None, vocab_size])

X = x
Y = y

l1 = LSTM(input_shape=(3, 1, 1), size=256)
l2 = LSTM(input_shape=(3, 1, 256), size=256, return_sequences=False)
l3 = Dense(input_shape=256, size=112)

layers = [l1, l2, l3]
model = Model(layers=layers)

gvs = model.train(X=X, Y=Y)
train = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1.0).apply_gradients(grads_and_vars=gvs)

pred = model.predict(X=X)
correct = tf.equal(tf.argmax(pred,1), tf.argmax(Y,1))
correct_sum = tf.reduce_sum(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

#########################

sess = tf.InteractiveSession()
sess.run(init)

step = 0
offset = random.randint(0,n_input+1)
end_offset = n_input + 1

correct_total = 0

while step < training_iters:
    # Generate a minibatch. Add some randomness on selection process.
    if offset > (len(training_data)-end_offset):
        offset = random.randint(0, n_input+1)

    symbols_in_keys = [ [dictionary[ str(training_data[i])]] for i in range(offset, offset+n_input) ]
    # symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])
    symbols_in_keys = np.reshape(np.array(symbols_in_keys), [n_input, -1, 1])

    symbols_out_onehot = np.zeros([vocab_size], dtype=float)
    symbols_out_onehot[dictionary[str(training_data[offset+n_input])]] = 1.0
    symbols_out_onehot = np.reshape(symbols_out_onehot,[1,-1])

    _, _correct_sum = sess.run([train, correct_sum], feed_dict={x: symbols_in_keys, y: symbols_out_onehot})
    correct_total += _correct_sum
    
    step += 1
    offset += (n_input+1)
    
    if (step % 1000) == 0:
        acc = correct_total / 1000.0
        print ('%d/%d: %f' % (step, training_iters, acc))
        correct_total = 0



