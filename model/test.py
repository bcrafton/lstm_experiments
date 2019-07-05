
import numpy as np

from Layer import Layer 
from LSTM import LSTM 
from Dense import Dense

inputs = np.random.uniform(size=(3, 1, 1))

l1 = LSTM(input_shape=(3, 1, 1), size=256)
l2 = LSTM(input_shape=(3, 1, 256), size=256)
l3 = Dense(input_shape=256, size=112)

out1, cache1 = l1.forward(inputs)
out2, cache2 = l2.forward(out1)
out3 = l3.forward(out2[-1])

###############################

e = out3 - np.random.rand(112)

###############################

back3 = l3.backward(out2[-1], out3, e)
back3_1 = np.zeros(shape=(1, 256))
back3_2 = np.zeros(shape=(1, 256))
back3_3 = back3
back3 = np.stack((back3_1, back3_2, back3_3), axis=0)

back2 = l2.backward(out1, out2, back3, cache1)
back1 = l1.backward(inputs, out1, back2, cache1)

