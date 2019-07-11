
import numpy as np

class Activation(object):
    def forward(self, x):
        assert(False)

    def gradient(self, x):
        assert(False)
        
class Sigmoid(Activation):
    def __init__(self):
        pass

    def forward(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def gradient(self, x):
        return x * (1 - x)
        
class Relu(Activation):

    def __init__(self):
        pass

    def forward(self, x):
        return (x > 0.0) * x

    def gradient(self, x):
        # pretty sure this gradient works for A and Z
        return (x > 0.0)

class Tanh(Activation):
# https://theclevermachine.wordpress.com/tag/tanh-function/ 

    def __init__(self):
        pass

    def forward(self, x):
        return np.tanh(x)

    def gradient(self, x):
        # this is gradient wtf A, not Z
        return 1 - np.power(x, 2)
        
class Linear(Activation):

    def __init__(self):
        pass

    def forward(self, x):
        return x 

    def gradient(self, x):
        return np.ones_like(x)
       
        
        
        
        
