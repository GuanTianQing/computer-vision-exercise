import numpy as np


def pool_forward(A_prev, hparameters, mode="max"):
    """
    实现池化层正向传播算法

    Arguments:
    A_prev -- 输入数据，numpy多维数组(m, n_H_prev, n_W_prev, n_C_prev)
              m：样本图像数量，n_H_prev：前一层图像高度，n_W_prev：前一层图像宽度，n_C_prev：前一层通道数量
    hparameters -- python字典，键值包含"f"和"stride"，对应的value分别表示池化窗口大小和池化步长，是池化的超参数
    mode -- 池化模式，字符串类型("max"或"average"，分别表示最大池化和平均池化)

    Returns:
    A -- 池化层输出，numpy多维数组(m, n_H, n_W, n_C)
         m：样本图像数量，n_H：本层图像高度，n_W：本层图像宽度，n_C：本层通道数量
    cache -- 保存池化层反向传播所需各种数值的缓存，包括A_prev（输入数据）和hparameters（超级参数）
    """

    # 从输入数据提取额维度
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    # 从"hparameters"提取超级参数
    f = hparameters["f"]
    stride = hparameters["stride"]

    # 计算输出维度
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev

    # 初始化4维输出数组A
    A = np.zeros((m, n_H, n_W, n_C))

    # ****** 在此开始编码 ****** #
   ##
    for i in range(m):                         # loop over the training examples
        for h in range(n_H):                     # loop on the vertical axis of the output volume
            for w in range(n_W):                 # loop on the horizontal axis of the output volume
                for c in range (n_C):            # loop over the channels of the output volume
                    
                    # Find the corners of the current "slice" (≈4 lines)
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                    
                    # Use the corners to define the current slice on the ith training example of A_prev, channel c. (≈1 line)
                    a_prev_slice = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]
                    
                    # Compute the pooling operation on the slice. Use an if statment to differentiate the modes. Use np.max/np.mean.
                    if mode == "max":
                        A[i, h, w, c] = np.max(a_prev_slice)
                    elif mode == "average":
                        A[i, h, w, c] = np.mean(a_prev_slice)

    # ****** 在此开始编码 ****** #

    # 在"cache"保存输入层（上一层，A_prev）和超级参数（hparameters）以便反向传播函数pool_backward()使用
    cache = (A_prev, hparameters)

    # 验证输出维度是否正确
    assert (A.shape == (m, n_H, n_W, n_C))

    return A, cache


# 测试函数
def test():
    print("\n")
    print("4. pool_forward()函数测试结果：")
    print("**********************************")

    np.random.seed(1)
    A_prev = np.random.randn(2, 4, 4, 3)
    hparameters = {"stride": 2, "f": 3}

    A, cache = pool_forward(A_prev, hparameters)
    print("mode = max")
    print("A =", A)
    print()
    A, cache = pool_forward(A_prev, hparameters, mode="average")
    print("mode = average")
    print("A =", A)
    print("**********************************")


# 运行测试
if __name__ == "__main__":
    test()
