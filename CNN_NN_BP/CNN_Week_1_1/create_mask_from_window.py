import numpy as np


def create_mask_from_window(x):
    """
    根据输入矩阵x创建一个掩蔽矩阵（mask matrix），用来标志x矩阵的最大元素位置

    Arguments:
    x -- 维度为(f, f)的numpy数组

    Returns:
    mask -- 与窗口同维度的数组，在对应x的最大元素位置的置为True
    """

    # ****** 在此开始编码 ****** #
    mask = (x == np.max(x))
    # ****** 在此开始编码 ****** #

    return mask


# 测试函数
def test():
    print("\n")
    print("6. create_mask_from_window()函数测试结果：")
    print("**********************************")
    np.random.seed(1)
    x = np.random.randn(2, 3)
    mask = create_mask_from_window(x)
    print('x = ', x)
    print("mask = ", mask)
    print("**********************************")


# 运行测试
if __name__ == "__main__":
    test()
