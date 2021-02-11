import numpy as np


def distribute_value(dz, shape):
    """
    将输入值dz平均分配到维度为shape的矩阵个元素之中

    Arguments:
    dz -- 输入的标量
    shape -- 输出矩阵的维度(n_H, n_W)

    Returns:
    a -- 维度为(n_H, n_W)的numpy数组，其各个元素的值是dz的平均分配
    """

    # ****** 在此开始编码 ****** #
    # 从shape提取维度 (≈1 line)
    (n_H, n_W) = None

    # 计算分配到矩阵各元素的值 (≈1 line)
    average =  dz / (n_H * n_W)

    # 创建矩阵，保证其每个元素的值为average所代表的值(≈1 line)
    a = np.ones(shape) * average
    # ****** 在此结束编码 ****** #

    return a


# 测试函数
def test():
    print("\n")
    print("7. distribute_value()函数测试结果：")
    print("**********************************")
    a = distribute_value(2, (2, 2))
    print('distributed value =', a)
    print("**********************************")


# 运行测试
if __name__ == "__main__":
    test()
