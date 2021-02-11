# -*- coding: utf-8 -*-
import numpy as np

class Softmax(object):
    
    def __init__(self):
        pass
        
        
    def forward(self, X):
        shiftX = X - np.max(X, axis=1).reshape((X.shape[0], -1)) # for numerical stability
        # 任务2。利用上文的shiftX计算softmax的输出p
        p = np.zeros_like(shiftX)
        for i in range(shiftX.shape[0]):
            #向量化操作，求样本各类别归一化后的概率值
            p[i,:] = np.exp(shiftX[i,:])/sum(np.exp(shiftX[i,:]))#向量/标量
        
        return p
    
    
    def backward(self, err_in):
        return err_in#

