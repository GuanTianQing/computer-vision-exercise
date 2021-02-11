# -*- coding: utf-8 -*-
import numpy as np

class Linear(object):
    
    def __init__(self, in_size, out_size, lr):
        self.W = np.random.normal(0, 0.01, (in_size+1, out_size))
        self.X = None
        self.lr = lr
        self.mmt = 0.95 # momentum
        self.v = np.zeros_like(self.W) # velocity for momentum update
        
        
    def forward(self, X):
        self.X = np.hstack( (np.ones((X.shape[0], 1)), X) )
        z = np.matmul(self.X, self.W)
        return z
    
    
    def backward(self, err_in):
        err_out = np.matmul(err_in, np.transpose(self.W))#更新残差
        dW = np.matmul(np.transpose(self.X), err_in)#残差乘以对应的输入向量，得到梯度
        
        # gradient descent update
        # self.W = self.W - self.lr * dW
        
        # momentum update
        self.v = self.v * self.mmt - self.lr * dW
        self.W = self.W + self.v 
        
        # NAG
        #v_bak = self.v.copy()
        #self.v = self.v * self.mmt - self.lr * dW
        #self.W = self.W - self.mmt * v_bak + (1 + self.mmt) * self.v
        return err_out[:, 1:]