import numpy as np
from zero_pad import zero_pad
from conv_single_step import conv_single_step


# 正向传播算法（正向卷积）
def conv_forward(A_prev, W, b, hparameters):
    """
    实现卷积函数的正向传播算法

    参数:
    A_prev -- 上一层输出的激活值，numpy多维数组(m, n_H_prev, n_W_prev, n_C_prev)
              m：样本图像数量，n_H_prev：前一层图像高度，n_W_prev：前一层图像宽度，n_C_prev：前一层通道数量
    W -- 权重，numpy多维数组(f, f, n_C_prev, n_C)
         f：卷积核的高度，也是卷积核的宽度，n_C_prev：前一层通道数量，n_C：本层卷积核数量
    b -- 偏置，numpy多维数组(1, 1, 1, n_C)，n_C：本层卷积核数量
    hparameters -- python字典，键值包含"stride"和"pad"，对应的value分别表示卷积步长和0填充个数，是卷积的超参数

    返回:
    Z -- 卷积输出，numpy多维数组(m, n_H, n_W, n_C)
         m：样本图像数量，n_H：本层图像高度，n_W：本层图像宽度，n_C：本层通道数量
    cache -- conv_backward()所需各种数值的缓存
    """

    # ****** 在此开始编码 ****** #
    # 从A_prev多维数组提取维度(≈1行代码)
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    # 从W多维数组提取维度 (≈1行代码)
    (f, f, n_C_prev, n_C) = W.shape

    # 从"hparameters"提取所需信息（超级参数）(≈2行代码)
    stride = hparameters['stride']
    pad = hparameters['pad']

    # 计算卷积输出卷的维度，提示: 使用int()实现下取整(≈2行代码)
    n_H = 1 + int((n_H_prev + 2 * pad - f) / stride)
    n_W = 1 + int((n_W_prev + 2 * pad - f) / stride)

    # 用zeros初始化输出卷Z(≈1行代码)
    Z = np.zeros((m, n_H, n_W, n_C))

    # 通过填充A_prev来创建A_prev_pad
    A_prev_pad = zero_pad(A_prev, pad)

    for i in range(m):  # 循环遍历全部训练样本
        a_prev_pad = A_prev_pad[i]   # 选择第i个训练样本填充后的激活值
        for h in range(n_H):  # 循环遍历输出卷的垂直轴
            for w in range(n_W):  # 循环遍历输出卷的水平轴
                for c in range(n_C):  # 循环遍历输出卷的通道(=滤器或卷积核的数量)

                    # 定位当前切片（slice）的位置(≈4行代码)
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start =  w * stride
                    horiz_end =  horiz_start + f

                    # 利用左上角位置从a_prev_pad生产当前切片（3维）(≈1行代码)
                    a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                    # 用卷积核W和偏置b对当前切片（3维）进行卷积，得到一个神经元输出(≈1行代码)
                    Z[i, h, w, c] = np.sum(np.multiply(a_slice_prev, W[:, :, :, c]) + b[:, :, :, c])

    # ****** 在此开始编码 ****** #

    # 验证输出维度是否正确
    assert (Z.shape == (m, n_H, n_W, n_C))

    # 在"cache"中保存反向传播所需的信息
    cache = (A_prev, W, b, hparameters)

    return Z, cache


# 测试函数
def test():
    print("\n")
    print("3. conv_forward()函数测试结果：")
    print("**********************************")

    # 准备参数
    np.random.seed(1)
    A_prev = np.random.randn(10, 4, 4, 3)
    W = np.random.randn(2, 2, 3, 8)
    b = np.random.randn(1, 1, 1, 8)
    hparameters = {"pad": 2,
                   "stride": 2}

    # 调用conv_forward()函数
    Z, cache_conv = conv_forward(A_prev, W, b, hparameters)

    # 输出结果
    print("Z的均值 =", np.mean(Z))
    print("Z[3,2,1] =", Z[3, 2, 1])
    print("cache_conv[0][1][2][3] =", cache_conv[0][1][2][3])
    print("**********************************")


# 运行测试
if __name__ == "__main__":
    test()
