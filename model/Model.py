
import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=1000)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

class Model:
    def __init__(self, layers):
        self.num_layers = len(layers)
        self.layers = layers
        
    def train(self, X, Y):
        A = [None] * self.num_layers
        C = [None] * self.num_layers
        D = [None] * self.num_layers
        grads_and_vars = []
        
        for ii in range(self.num_layers):
            l = self.layers[ii]
            if ii == 0:
                A[ii], C[ii] = l.forward(X)
            else:
                A[ii], C[ii] = l.forward(A[ii-1])

        # [T, B, N_in]
        N = np.shape(X)[1]
        E = (softmax(A[self.num_layers-1]) - Y) / N
        correct = np.argmax(A[self.num_layers-1], axis=1) == np.argmax(Y, axis=1)
        correct_sum = np.sum(correct)

        for ii in range(self.num_layers-1, -1, -1):
            l = self.layers[ii]
            
            if (ii == self.num_layers-1):
                D[ii] = l.backward(A[ii-1], A[ii], E, C[ii])
            elif (ii == 0):
                D[ii] = l.backward(X, A[ii], D[ii+1], C[ii])
            else:
                D[ii] = l.backward(A[ii-1], A[ii], D[ii+1], C[ii])
                
        return correct_sum
              
    def predict(self, X):
        A = [None] * self.num_layers
        
        for ii in range(self.num_layers):
            l = self.layers[ii]
            if ii == 0:
                A[ii], _ = l.forward(X)
            else:
                A[ii], _ = l.forward(A[ii-1])
                
        return A[self.num_layers-1]
        
        
        
        
        
        
        
        
