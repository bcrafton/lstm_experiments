
import numpy as np

from Layer import Layer 
from LSTM import LSTM 
from Dense import Dense
from Model import Model

###############################

Wa_x = np.array([[0.45, 0.25]])
Wi_x = np.array([[0.95, 0.80]])
Wf_x = np.array([[0.70, 0.45]])
Wo_x = np.array([[0.60, 0.40]])

Wa_h = np.array([0.15])
Wi_h = np.array([0.8])
Wf_h = np.array([0.1])
Wo_h = np.array([0.25])

ba = np.array([0.20])
bi = np.array([0.65])
bf = np.array([0.15])
bo = np.array([0.10])

x0 = np.array([1.00, 2.00])
y0 = 0.5

x1 = np.array([0.50, 3.00])
y1 = 1.25

###############################

x = np.array([[x0], [x1]]); # print ('x', np.shape(x))
y = np.array([[y0], [y1]]); # print ('y', np.shape(y))

l1 = LSTM(input_shape=(2, 1, 2), size=1)
out, cache = l1.forward(X=x)

out = np.reshape(out, [2, 1])
do = out - y

di = l1.backward(AI=x, AO=out, DO=do, cache=cache)

# print ('di')
# print (di)

