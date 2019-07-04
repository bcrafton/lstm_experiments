
import tensorflow as tf
import numpy as np

from Layer import Layer 
from LSTM import LSTM 

inputs = np.random.uniform(size=(3, 1, 1))

l1 = LSTM(input_shape=(3, 1, 1), size=256)
l2 = LSTM(input_shape=(3, 1, 256), size=256)

out1, states1 = l1.forward(inputs)
out2, states2 = l2.forward(out1)
