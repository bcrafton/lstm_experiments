
import numpy as np

from Layer import Layer 
from LSTM import LSTM 
from Dense import Dense
from Model import Model

inputs = np.random.uniform(size=(3, 1, 1))
labels = np.random.uniform(size=112)

l1 = LSTM(input_shape=(3, 1, 1), size=256)
l2 = LSTM(input_shape=(3, 1, 256), size=256, return_sequences=False)
l3 = Dense(input_shape=256, size=112)

layers = [l1, l2, l3]
model = Model(layers=layers)

out = model.predict(X=inputs)
model.train(X=inputs, Y=labels)
