
import tensorflow as tf
import numpy as np

from Layer import Layer 
from LSTM import LSTM 
from Dense import Dense

X = tf.placeholder(tf.float32, [3, 1, 1])

l1 = LSTM(input_shape=(3, 1, 1), size=256)
l2 = LSTM(input_shape=(3, 1, 256), size=256, return_sequences=False)
l3 = Dense(input_shape=256, size=112)

out1, cache1 = l1.forward(X)
out2, cache2 = l2.forward(out1)
out3, _ = l3.forward(out2)

init = tf.global_variables_initializer()

####################

sess = tf.InteractiveSession()
sess.run(init)

inputs = np.random.uniform(size=(3, 1, 1))
[out1, out2, out3] = sess.run([out1, out2, out3], feed_dict={X: inputs})

print (np.shape(out1), np.shape(out2), np.shape(out3))
