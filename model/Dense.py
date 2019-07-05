
import numpy as np

from Layer import Layer 
from Activation import Activation
from Activation import Linear

from init_matrix import init_matrix

class Dense(Layer):

    def __init__(self, input_shape, size, activation=None, init=None, lr=0.001):
    
        self.input_shape = input_shape
        self.size = size
        self.init = init
        self.activation = Linear() if activation == None else activation
        self.lr = lr

        self.bias = np.zeros(shape=size)
        self.weights = init_matrix(size=(self.input_shape, self.size), init=self.init)

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

        DW = AI.T @ DO
        DB = np.sum(DO, axis=0)
        
        self.weights -= self.lr * DW
        self.bias -= self.lr * DB
        
        return DI
        
    ###################################################################

        
        
