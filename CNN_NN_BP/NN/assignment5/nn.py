# -*- coding: utf-8 -*-

class NeuralNetwork(object):
    
    def __init__(self):
        self.layers = []
        self.h = None
        self.err_out = None
        
        
    def forward(self, X):
        self.h = X
        for layer in self.layers:
            self.h = layer.forward(self.h)
        return self.h
            
            
    def backward(self, err_in):
        self.err_out = err_in
        #从后往前更新
        for i in range(len(self.layers)-1, -1, -1):
            self.err_out = self.layers[i].backward(self.err_out)
