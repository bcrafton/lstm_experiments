
import numpy as np

from Layer import Layer 
from Activation import Activation
from Activation import Linear

class Dense(Layer):

    def __init__(self, input_shape, size, activation=None, init_weights='sqrt_fan_in'):
    
        self.input_shape = input_shape
        self.size = size
        self.activation = Linear() if activation == None else activation

        self.bias = np.zeros(shape=size)

        if init_weights == 'zero':
            self.weights = np.zeros(shape=(self.input_shape, self.size))
        elif init_weights == 'sqrt_fan_in':
            sqrt_fan_in = np.sqrt(self.input_shape)
            self.weights = np.random.uniform(low=-1.0/sqrt_fan_in, high=1.0/sqrt_fan_in, size=(self.input_shape, self.size))
        elif init_weights == 'alexnet':
            self.weights = np.random.normal(loc=0.0, scale=0.01, size=(self.input_shape, self.size))
        else:
            # glorot
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
        return DI
        
    def train(self, AI, AO, DO, cache):
        if not self._train:
            return []
            
        batch, _ = np.shape(AI)
        
        DO = DO * self.activation.gradient(AO)
        DW = AI.T @ DO
        DB = np.sum(DO, axis=0)

        return [(DW, self.weights), (DB, self.bias)]

    ###################################################################

        
        
