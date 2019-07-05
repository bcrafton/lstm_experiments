
import numpy as np

from Layer import Layer 
from Activation import Activation
from Activation import Linear

class Dense(Layer):

    def __init__(self, input_shape, size, activation=None, init=None, lr=0.001):
    
        self.input_shape = input_shape
        self.size = size
        self.init = init
        self.activation = Linear() if activation == None else activation
        self.lr = lr

        self.bias = np.zeros(shape=size)

        if self.init == 'zero':
            self.weights = np.zeros(shape=(self.input_shape, self.size))
        elif self.init == 'sqrt_fan_in':
            sqrt_fan_in = np.sqrt(self.input_shape)
            self.weights = np.random.uniform(low=-1.0/sqrt_fan_in, high=1.0/sqrt_fan_in, size=(self.input_shape, self.size))
        elif self.init == 'glorot_uniform':
            high = np.sqrt(6. / (self.input_shape + self.size))
            low = -high
            self.weights = np.random.uniform(low=low, high=high, size=(self.input_shape, self.size))
        elif self.init == 'glorot_normal':
            scale = np.sqrt(2. / (self.input_shape + self.size))
            self.weights = np.random.normal(loc=0.0, scale=scale, size=(self.input_shape, self.size))
        elif self.init == None:
            self.weights = np.random.normal(loc=0.0, scale=1., size=(self.input_shape, self.size))
        else:
            assert(False)

    ###################################################################
        
    def get_weights(self):
        assert (False)

    def num_params(self):
        assert (False)

    ###################################################################

    def forward(self, X):
        Z = X @ self.weights + self.bias
        A = self.activation.forward(Z)
        return A, None
            
    def backward(self, AI, AO, DO, cache):
        DO = DO * self.activation.gradient(AO)
        DI = DO @ self.weights.T
        
        DO = DO * self.activation.gradient(AO)
        DW = AI.T @ DO
        DB = np.sum(DO, axis=0)
        
        self.weights -= self.lr * DW
        self.bias -= self.lr * DB
        
        return DI
        
    ###################################################################

        
        
