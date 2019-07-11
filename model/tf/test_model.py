
import tensorflow as tf
import numpy as np

from Layer import Layer 
from LSTM import LSTM 
from Dense import Dense
from Model import Model

################

X = tf.placeholder(tf.float32, [3, 1, 1])
Y = tf.placeholder(tf.float32, [1, 112])

l1 = LSTM(input_shape=(3, 1, 1), size=256)
l2 = LSTM(input_shape=(3, 1, 256), size=256, return_sequences=False)
l3 = Dense(input_shape=256, size=112)

layers = [l1, l2, l3]
model = Model(layers=layers)

pred = model.predict(X=X)
gvs = model.train(X=X, Y=Y)

init = tf.global_variables_initializer()

################

sess = tf.InteractiveSession()
sess.run(init)

inputs = np.random.uniform(size=(3, 1, 1))
labels = np.random.uniform(size=(1, 112))

[pred, gvs] = sess.run([pred, gvs], feed_dict={X: inputs, Y: labels})

print (np.shape(pred))
for gv in gvs:
    print (np.shape(gv))

