import numpy as np
from create_mask_from_window import create_mask_from_window
from distribute_value import distribute_value
from pool_forward import pool_forward


def pool_backward(dA, cache, mode="max"):
    """
    实现池化层的反向传播算法

    Arguments:
    dA -- 针对池化层输出的损失梯度，维度与A相同
    cache -- 池化层正向传播的缓存输出，包括该层的输入数据和hparameters（超级参数）
    mode -- 池化模式，字符串类型("max"或"average"，分别表示最大池化和平均池化)

    Returns:
    dA_prev -- 针对池化层输入数据损失梯度，维度与A_prev相同
    """

    ### 在此开始编码 ###

    # 从cache提取信息 (≈1 line)
    (A_prev, hparameters) = cache

    # 从"hparameters"提取超级参数 (≈2 lines)
    stride = hparameters['stride']
    f =  hparameters['f']

    # 从A_prev和dA提取维度(≈2 lines)
    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    m, n_H, n_W, n_C = dA.shape

    # 用0初始化dA_prev(≈1 line)
    dA_prev = np.zeros_like(A_prev)
    
    for i in range(None):  # 循环遍历全部训练样本

        # 从A_prev选择第i个训练样本的激活值 (≈1 line)
        a_prev = A_prev[i]

        for h in range(n_H):  # 循环遍历垂直轴
            for w in range(n_W):  # 循环遍历水平轴
                for c in range(n_C):  # 循环遍历通道(深度)

                    # 定位当前切片（slice）的位置(≈4 lines)
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f

                    # 计算两种模式的反向传播
                    if mode == "max":
                        # 利用左上角位置从a_prev的第i个训练样本的通道c产生当前切片(≈1 line)
                        a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                        # 从a_prev_slice创建掩蔽矩阵(≈1 line)
                        mask = create_mask_from_window(a_prev_slice)
                        # 将dA_prev设置为dA_prev + (掩蔽矩阵乘以dA的正确位置) (≈1 line)
                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += mask * dA[i, vert_start, horiz_start, c]

                    elif mode == "average":

                        # 从dA取出当前位置的激活值 (≈1 line)
                        da = dA[i, vert_start, horiz_start, c]
                        # 定义池化核的维度为fxf (≈1 line)
                        shape = (f, f)
                        # 通过均值分配来计算dA_prev的正确切片，即：将da的分配矩阵加到对应的切片位置(≈1 line)
                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += distribute_value(da, shape)

    ### 在此结束编码 ###

    # 验证输出维度是否正确
    assert (dA_prev.shape == A_prev.shape)

    return dA_prev


# 测试函数
def test():
    print("\n")
    print("8. pool_backward()函数测试结果：")
    print("**********************************")

    np.random.seed(1)
    A_prev = np.random.randn(5, 5, 3, 2)
    hparameters = {"stride": 1, "f": 2}
    A, cache = pool_forward(A_prev, hparameters)
    dA = np.random.randn(5, 4, 2, 2)

    dA_prev = pool_backward(dA, cache, mode="max")
    print("mode = max")
    print('mean of dA = ', np.mean(dA))
    print('dA_prev[1,1] = ', dA_prev[1, 1])
    print()
    dA_prev = pool_backward(dA, cache, mode="average")
    print("mode = average")
    print('mean of dA = ', np.mean(dA))
    print('dA_prev[1,1] = ', dA_prev[1, 1])
    print("**********************************")


# 运行测试
if __name__ == "__main__":
    test()
