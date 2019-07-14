
import tensorflow as tf
import numpy as np

from Layer import Layer 
from LSTM import LSTM

###############################

X = tf.placeholder(tf.float32, [1, 2, 2])
Y = tf.placeholder(tf.float32, [1, 2, 1])

l1 = LSTM(input_shape=(1, 2, 2), size=1)
out, cache = l1.forward(X=X)

do = out - Y

di, dt, dw = l1.backward(AI=X, AO=out, DO=do, cache=cache)

init = tf.global_variables_initializer()

###############################

x0 = np.array([1.00, 2.00])
y0 = 0.5

x1 = np.array([0.50, 3.00])
y1 = 1.25

x = np.array([x0, x1]).reshape(1, 2, 2)
y = np.array([y0, y1]).reshape(1, 2, 1)

x_t = np.transpose(x, (1, 0, 2))

###############################

sess = tf.InteractiveSession()
sess.run(init)

[out, di, dt, dw] = sess.run([out, di, dt, dw], feed_dict={X: x, Y: y})

'''
print (np.shape(out))
print (np.shape(di))
print (np.shape(dt))
'''
'''
print (x_t[0])
print (x_t[1])
'''

print (out)
print (di)
print (dt)

'''
# print (np.shape(dw))
for (g, v) in dw:
    print (g)
'''
