# -*- coding: utf-8 -*-
import numpy as np

class Sigmoid(object):
    
    def __init__(self):
        pass
        
        
    def forward(self, X):
        self.activation = 1 / (np.exp(-X) + 1)
        return self.activation
            
            
    def backward(self, err_in):
        return err_in * self.activation * (1 - self.activation)
    
        
        