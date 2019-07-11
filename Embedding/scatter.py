
import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=np.inf)

#########################################################

w_np = np.random.uniform(size=(100, 10))
idx_np = np.random.randint(low=0, high=100, size=(5,))
update_np = np.ones(shape=(5, 10))

w = tf.Variable(w_np, dtype=tf.float32)
idx = tf.Variable(idx_np, dtype=tf.int32)
update = tf.Variable(update_np, dtype=tf.float32)

w = tf.scatter_update(w, idx, update)
'''
idx = tf.reshape(idx, [5, 1])
vec = tf.gather_nd(w, idx)
'''
vec = tf.gather(w, idx)

#########################################################

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.local_variables_initializer().run()

#####################################################

[vec] = sess.run([vec], feed_dict={})
print (vec)
print ()
print (w_np[idx_np])

#####################################################






