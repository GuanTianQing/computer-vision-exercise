# -*- coding: utf-8 -*-
import numpy as np

class Relu(object):
    
    def __init__(self):
        self.X = None
        
    # 任务1。根据该函数的表达式计算函数值，提示：利用 np.maximum 方法。     
    def forward(self, X):
        self.x = X
         #大于零的保留，小于零的置为零
        self.activation = np.maximum(0, X)
        return self.activation    
                   
            
    def backward(self, err_in):
        # 任务1。计算函数输出对输入的导数dfdX
        dfdX = self.activation
        #输入向量x为非零值的导数为一，否则为零
        dfdX[np.nonzero(self.x)] = 1
        
        err_out = dfdX*err_in#更新残差
                
        return err_out
    
        
        