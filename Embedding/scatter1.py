import tensorflow as tf
import numpy as np

init = [[0, 0, 0, 0],[0, 0, 0, 0]]
idx = [0, 1]
updates = [[1, 0, 0, 0], [1, 0, 0, 0]]

g = tf.Graph()
with g.as_default():
    a = tf.Variable(initial_value=init)
    b = tf.scatter_update(a, idx, updates)

with tf.Session(graph=g) as sess:
   sess.run(tf.initialize_all_variables())
   print (sess.run(a))
   print (sess.run(b))
   
print (np.shape(init))
print (np.shape(idx))
print (np.shape(updates))
