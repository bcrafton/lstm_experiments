
import tensorflow as tf
import numpy as np
import math

from Layer import Layer 

########################################

def tanh(x):
    return np.tanh(x)

def dtanh(x):
    # wrt A, not Z
    return 1 - np.power(x, 2)
    
    # wrt Z, not A
    # return 1. - np.tanh(x) ** 2.

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def dsigmoid(x):
    # wrt A, not Z
    return x * (1 - x)
    
########################################

class LSTM(Layer):

    def __init__(self, input_shape, size, lr=0.001, return_sequences=True):
        self.input_shape = input_shape
        self.time_size, self.batch_size, self.input_size = self.input_shape
        self.output_size = size
        self.lr = lr
        self.return_sequences = return_sequences
        
        '''
        self.Wa_x = np.random.normal(loc=0.0, scale=0.01, size=(self.input_size, self.output_size))
        self.Wi_x = np.random.normal(loc=0.0, scale=0.01, size=(self.input_size, self.output_size))
        self.Wf_x = np.random.normal(loc=0.0, scale=0.01, size=(self.input_size, self.output_size))
        self.Wo_x = np.random.normal(loc=0.0, scale=0.01, size=(self.input_size, self.output_size))

        self.Wa_h = np.random.normal(loc=0.0, scale=0.01, size=(self.output_size, self.output_size))
        self.Wi_h = np.random.normal(loc=0.0, scale=0.01, size=(self.output_size, self.output_size))
        self.Wf_h = np.random.normal(loc=0.0, scale=0.01, size=(self.output_size, self.output_size))
        self.Wo_h = np.random.normal(loc=0.0, scale=0.01, size=(self.output_size, self.output_size))

        self.ba = np.zeros(shape=self.output_size)
        self.bi = np.zeros(shape=self.output_size)
        self.bf = np.zeros(shape=self.output_size)
        self.bo = np.zeros(shape=self.output_size)
        '''
        
        self.Wa_x = np.array([[0.45], [0.25]])
        self.Wi_x = np.array([[0.95], [0.80]])
        self.Wf_x = np.array([[0.70], [0.45]])
        self.Wo_x = np.array([[0.60], [0.40]])

        self.Wa_h = np.array([[0.15]])
        self.Wi_h = np.array([[0.8]])
        self.Wf_h = np.array([[0.1]])
        self.Wo_h = np.array([[0.25]])

        self.ba = np.array([0.20])
        self.bi = np.array([0.65])
        self.bf = np.array([0.15])
        self.bo = np.array([0.10])

    ###################################################################
        
    def get_weights(self):
        assert (False)

    def num_params(self):
        assert (False)

    ###################################################################

    def forward(self, X):
        if (np.shape(X) != (self.time_size, self.batch_size, self.input_size)):
            print (np.shape(X))
            assert(np.shape(X) == (self.time_size, self.batch_size, self.input_size))
    
        la = []
        li = []
        lf = []
        lo = []
        ls = []
        lh = []
        
        for t in range(self.time_size):
            x = X[t]
            
            if t == 0:
                a = tanh(   x @ self.Wa_x + self.ba) 
                i = sigmoid(x @ self.Wi_x + self.bi) 
                f = sigmoid(x @ self.Wf_x + self.bf)
                o = sigmoid(x @ self.Wo_x + self.bo) 
            else:
                a = tanh(   x @ self.Wa_x + lh[t-1] @ self.Wa_h + self.ba)
                i = sigmoid(x @ self.Wi_x + lh[t-1] @ self.Wi_h + self.bi)
                f = sigmoid(x @ self.Wf_x + lh[t-1] @ self.Wf_h + self.bf)
                o = sigmoid(x @ self.Wo_x + lh[t-1] @ self.Wo_h + self.bo)

            la.append(a)
            li.append(i)
            lf.append(f)
            lo.append(o)
            
            if t == 0:
                s = a * i               
                ls.append(s)
            else:
                s = a * i + ls[t-1] * f
                ls.append(s)
                
            h = tanh(s) * o
            lh.append(h)

        # [T, B, O]
        outputs = np.stack(lh, axis=0)

        cache = {}
        cache['a'] = la
        cache['i'] = li
        cache['f'] = lf
        cache['o'] = lo
        cache['s'] = ls
        cache['h'] = lh
        
        if self.return_sequences:
            return outputs, cache
        else:
            return outputs[-1], cache


    # combining backward and train together
    def backward(self, AI, AO, DO, cache):
        if not self.return_sequences:
            reshape_DO = np.zeros(shape=(self.time_size, self.batch_size, self.output_size))
            reshape_DO[-1] = DO
            DO = reshape_DO
    
        a = cache['a'] 
        i = cache['i'] 
        f = cache['f'] 
        o = cache['o'] 
        s = cache['s'] 
        h = cache['h'] 
        
        lds = [None] * self.time_size
        ldx = []
        
        dWa_x = np.zeros_like(self.Wa_x)
        dWi_x = np.zeros_like(self.Wi_x)
        dWf_x = np.zeros_like(self.Wf_x)
        dWo_x = np.zeros_like(self.Wo_x)

        dWa_h = np.zeros_like(self.Wa_h)
        dWi_h = np.zeros_like(self.Wi_h)
        dWf_h = np.zeros_like(self.Wf_h)
        dWo_h = np.zeros_like(self.Wo_h)

        for t in range(self.time_size-1, -1, -1):
            if t == 0:
                dh = DO[t] + dout
                ds = dh * o[t] * dtanh(tanh(s[t])) + lds[t+1] * f[t+1]
                da = ds * i[t] * dtanh(a[t])
                di = ds * a[t] * dsigmoid(i[t]) 
                df = ds
                do = dh * tanh(s[t]) * dsigmoid(o[t]) 
                
                '''
                # it was f[t] not f[t+1]
                print (dh)
                print (o[t])
                print (dtanh(tanh(s[t])))
                print (lds[t+1])
                print (f[t+1])
                '''
                '''
                # broken bc of ds.
                print (ds)
                print (i[t])
                print (dtanh(a[t]))
                '''
                '''
                # all the broken
                print (da)
                print (di)
                print (df)
                print (do)
                print (AI[t].T)
                '''
            elif t == self.time_size-1:
                dh = DO[t]
                ds = dh * o[t] * dtanh(tanh(s[t]))
                da = ds * i[t] * dtanh(a[t])
                di = ds * a[t] * dsigmoid(i[t]) 
                df = ds * s[t-1] * dsigmoid(f[t]) 
                do = dh * tanh(s[t]) * dsigmoid(o[t])
            else:
                assert(False)
                dh = DO[t] + dout
                ds = dh * o[t] * dtanh(tanh(s[t])) + lds[t+1] * f[t]
                da = ds * i[t] * dtanh(a[t])
                di = ds * a[t] * dsigmoid(i[t]) 
                df = ds * s[t-1] * dsigmoid(f[t]) 
                do = dh * tanh(s[t]) * dsigmoid(o[t]) 

            dout_a = da @ self.Wa_h.T
            dout_i = di @ self.Wi_h.T
            dout_f = df @ self.Wf_h.T
            dout_o = do @ self.Wo_h.T
            dout = dout_a + dout_i + dout_f + dout_o
            
            dx_a = da @ self.Wa_x.T
            dx_i = di @ self.Wi_x.T
            dx_f = df @ self.Wf_x.T
            dx_o = do @ self.Wo_x.T
            dx = dx_a + dx_i + dx_f + dx_o
            
            dWa_x += AI[t].T @ da
            dWi_x += AI[t].T @ di 
            dWf_x += AI[t].T @ df 
            dWo_x += AI[t].T @ do 
            
            if t > 0:
                dWa_h += h[t-1].T @ da
                dWi_h += h[t-1].T @ di 
                dWf_h += h[t-1].T @ df 
                dWo_h += h[t-1].T @ do 
                
            lds[t] = ds
            ldx.append(dx)

        self.Wa_x -= self.lr * dWa_x
        self.Wi_x -= self.lr * dWi_x
        self.Wf_x -= self.lr * dWf_x
        self.Wo_x -= self.lr * dWo_x
        
        self.Wa_h -= self.lr * dWa_h
        self.Wi_h -= self.lr * dWi_h
        self.Wf_h -= self.lr * dWf_h
        self.Wo_h -= self.lr * dWo_h
        
        dWx = np.concatenate((dWa_x, dWi_x, dWf_x, dWo_x), axis=1)
        dWh = np.concatenate((dWa_h, dWi_h, dWf_h, dWo_h), axis=1)
        
        print (dWx.T)
        print (dWh.T)

        return ldx

    ###################################################################

        
