import numpy as np


# 单步卷积
def conv_single_step(a_slice_prev, W, b):
    """
    对前一层输出的激活值的单个切片（a_slice_prev）实施以参数W为核的卷积操作

    参数:
    a_slice_prev -- 单个切片多维数组(f, f, n_C_prev)，f：卷积核的高度，也是卷积核的宽度，n_C_prev：前一层通道数量
    W -- 卷积核窗口的权重参数 - 多维数组(f, f, n_C_prev)，f：卷积核的高度，也是卷积核的宽度，n_C_prev：前一层通道数量
    b -- 卷积核窗口的偏置参数 - 多维数组(1, 1, 1)

    返回值:
    Z -- 一个标量，结果是切片窗口(W, b)在切片a_slice_prev上的卷积
    """

    ### START CODE HERE ### (≈ 2 lines of code)
    # Element-wise product between a_slice and W. Add bias.
    s = np.multiply(a_slice_prev, W) 
    # Sum over all entries of the volume s
    Z = np.sum(s)+b
    ### END CODE HERE ###

    return Z


# 测试函数
def test():
    print("\n")
    print("2. conv_single_step()函数测试结果：")
    print("**********************************")
    np.random.seed(1)
    a_slice_prev = np.random.randn(4, 4, 3)
    W = np.random.randn(4, 4, 3)
    b = np.random.randn(1, 1, 1)

    Z = conv_single_step(a_slice_prev, W, b)
    print("Z =", Z)
    print("**********************************")


# 运行测试
if __name__ == "__main__":
    test()
