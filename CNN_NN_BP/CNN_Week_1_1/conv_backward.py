import numpy as np
from zero_pad import zero_pad
from conv_forward import conv_forward


def conv_backward(dZ, cache):
    """
    实现卷积函数的反向传播算法

    参数:
    dZ -- 针对卷积输出层(Z)的代价梯度，numpy多维数组(m, n_H, n_W, n_C)
          m：样本图像数量，n_H：图像高度，n_W：图像宽度，n_C：通道数量（卷积核数量）
    cache -- conv_backward()所需各种数值的缓存，是conv_forward()函数的输出

    返回:
    dA_prev -- 针对卷积层（A_prev）的代价梯度，numpy多维数组 (m, n_H_prev, n_W_prev, n_C_prev)
               m：样本图像数量，n_H_prev：前一层图像高度，n_W_prev：前一层图像宽度，n_C_prev：前一层通道数量
    dW -- 针对权重卷积层（W）的代价梯度，numpy多维数组(f, f, n_C_prev, n_C)
          f：卷积核的高度，也是卷积核的宽度，n_C_prev：前一层通道数量，n_C：本层卷积核数量
    db -- 针对偏置卷积层（b）的代价梯度，numpy多维数组(1, 1, 1, n_C)，n_C：本层卷积核数量
    """

    # ****** 在此开始编码 ****** #
    # 从缓存"cache"提取所需信息
    (A_prev, W, b, hparameters) = cache

    # 从A_prev多维数组提取维度
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    # 从W多维数组提取维度
    (f, f, n_C_prev, n_C) = W.shape

    # 从"hparameters"提取所需信息
    stride = hparameters['stride']
    pad = hparameters['pad']

    # 从dZ多维数组提取维度
    (m, n_H, n_W, n_C) = dZ.shape

    # 用正确的维度初始化dA_prev, dW, db
    dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))      
    dW = np.zeros((f, f, n_C_prev, n_C))
    db = np.zeros((1, 1, 1, n_C))

    # 填充A_prev和dA_prev
    A_prev_pad =  zero_pad(A_prev, pad)
    dA_prev_pad =  zero_pad(dA_prev, pad)

    for i in range(m):  # 循环遍历全部训练样本

        # 从A_prev_pad和dA_prev_pad选择第i个训练样本
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]#浅拷贝，da_prev_pad与dA_prev_pad[i]是同一个对象的不同名字

        for h in range(n_H):  # 循环遍历输出卷的垂直轴
            for w in range(n_W):  # 循环遍历输出卷的水平轴
                for c in range(n_C):  # 循环遍历输出卷的通道

                    # 定位当前切片（slice）的位置
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f

                    # 利用左上角位置从a_prev_pad生产切片
                    a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                    # 更新前一层激活值、卷积核、偏置的梯度
                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:,:,:,c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
                    db[:, :, :, c] += dZ[i, h, w, c]

        # 将第i个训练样本的dA_prev设置为去除填充的da_prev_pad（提示：请使用X[pad:-pad, pad:-pad, :]）
        dA_prev[i, :, :, :] = dA_prev_pad[i, pad:-pad, pad:-pad, :]
    # ****** 在此开始编码 ****** #

    # 验证输出维度是否正确
    assert (dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))

    return dA_prev, dW, db


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

    print("\n")
    print("5. conv_backward()函数测试结果：")
    print("**********************************")
    dA, dW, db = conv_backward(Z, cache_conv)
    print("dA_mean =", np.mean(dA))
    print("dW_mean =", np.mean(dW))
    print("db_mean =", np.mean(db))
    print("**********************************")


# 运行测试
if __name__ == "__main__":
    test()
