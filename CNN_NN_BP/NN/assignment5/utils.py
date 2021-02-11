# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np


def show_mnist(X, y, n_rows, n_cols):
    plt.figure()
    for i in range(n_rows*n_cols):
        im = X[i, :, :].reshape((28, 28))
        plt.subplot(n_rows, n_cols, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.title('Label: {0}'.format(y[i, 0]))
        plt.imshow(im, cmap='gray', interpolation='None')
    plt.show()
    
    
def make_onehot(y, K):   
  # 任务3：将y扩张成one-hot形式
  # y：label向量，K:多分类问题的类别个数   
    y_onehot = np.zeros( (y.size, K) )   
    #样本有类别属性，将样本所对应的类别属性置为1
    for i in range(y_onehot.shape[0]):
        y_onehot[i][y[i]] = 1
    
    return y_onehot


def cross_entropy(p, y):
    # 任务4：对照视频教程理解下列代码。注意：将最小值限制在1e-10是为了避免log计算结果出现溢出。
    return - np.sum(y * np.log(p.clip(min=1e-10))) / p.shape[0]
    

def plot_loss(trainloss, validloss):
    plt.figure()
    plt.plot(trainloss, 'b-', validloss, 'r-')
    #plt.xlim(0, n_iter)
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title('learning curve')
    plt.legend(['training loss', 'validation loss'])
    plt.show()
    

def confusion_matrix(y_hat, y_target, K):
    cmatrix = np.zeros((K, K), dtype=np.int32)
    for i in range(y_hat.size):
        cmatrix[y_hat[i] , y_target[i]] += 1
    return cmatrix