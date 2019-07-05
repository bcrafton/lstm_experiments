
import numpy as np

from Layer import Layer 
from LSTM import LSTM 
from Dense import Dense

inputs = np.random.uniform(size=(3, 1, 1))

l1 = LSTM(input_shape=(3, 1, 1), size=256)
l2 = LSTM(input_shape=(3, 1, 256), size=256, return_sequences=False)
l3 = Dense(input_shape=256, size=112)

out1, cache1 = l1.forward(inputs)
out2, cache2 = l2.forward(out1)
out3 = l3.forward(out2)

print (np.shape(out1))
print (np.shape(out2))
print (np.shape(out3))

###############################

e = out3 - np.random.rand(112)

###############################

back3 = l3.backward(out2, out3, e)
back2 = l2.backward(out1, out2, back3, cache2)
back1 = l1.backward(inputs, out1, back2, cache1)

print (np.shape(back3))
print (np.shape(back2))
print (np.shape(back1))
