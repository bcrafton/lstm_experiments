
import numpy as np

#####

def tanh(x):
    return np.tanh(x)

def dtanh(x):
    # this is gradient wrt A, not Z
    return 1 - np.power(x, 2)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def dsigmoid(x):
    return x * (1 - x)
    
#####

N = 2
M = 1

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

#####

s_prev = [0.0]
h_prev = [0.0]

a0 = tanh(   Wa_x @ x0 + Wa_h @ h_prev + ba)
i0 = sigmoid(Wi_x @ x0 + Wi_h @ h_prev + bi)
f0 = sigmoid(Wf_x @ x0 + Wf_h @ h_prev + bf)
o0 = sigmoid(Wo_x @ x0 + Wo_h @ h_prev + bo)
s0 = a0 * i0 + s_prev * f0
h0 = tanh(s0) * o0

print ('s0', s0, 'h0', h0)

s_prev = s0
h_prev = h0

a1 = tanh(   Wa_x @ x1 + Wa_h @ h_prev + ba)
i1 = sigmoid(Wi_x @ x1 + Wi_h @ h_prev + bi)
f1 = sigmoid(Wf_x @ x1 + Wf_h @ h_prev + bf)
o1 = sigmoid(Wo_x @ x1 + Wo_h @ h_prev + bo)
s1 = a1 * i1 + s_prev * f1
h1 = tanh(s1) * o1

print ('s1', s1, 'h1', h1)




