import numpy as np


def zero_pad(X, pad):
    """
    对数据集X的所有图像四周用0进行填充，填充只针对每个图像的高度和宽度，如图1所示。

    参数:
    X -- 为python numpy多维数组(m, n_H, n_W, n_C)，用来表示一批图像，m：图像数量，n_H：图像高度，n_W：图像宽度，n_C：图像通道数量
    pad -- 整数，表示在每个图像的四周垂直和水平维度填充增加的维度

    返回:
    X_pad -- 填充后的批量图像多维数组(m, n_H + 2*pad, n_W + 2*pad, n_C)
             m：图像数量，n_H + 2*pad：填充后的图像高度，n_W + 2*pad：填充后的图像宽度，n_C：图像通道数量
    """

    # ****** 在此开始编码 ****** #
    X_pad = np.pad(X, ((0, 0),(pad, pad),(pad, pad),(0, 0)), 'constant', constant_values=0)

    # ****** 在此开始编码 ****** #

    return X_pad


# 测试函数
def test():
    print("1. zero_pad()函数测试结果：")
    print("**********************************")
    np.random.seed(1)
    # x = np.random.randn(4, 3, 3, 2)
    x = np.random.randint(0, 256, size=(4, 3, 3, 2))
    x_pad = zero_pad(x, 2)
    print("x的维度 =", x.shape)
    print("x_pad的维度 =", x_pad.shape)

    print("\n")
    print("x的第一个样本的第一个通道：")
    print(x[0, :, :, 0])
    print("\n")
    print("x_pad的第一个样本的第一个通道：")
    print(x_pad[0, :, :, 0])
    print("**********************************")

    # 绘制第一个样本的第一个通道原始数据和填充后的数据
    import matplotlib.pyplot as plt

    plt.rcParams['figure.figsize'] = (6.0, 4.0)  # 设置缺省图像尺寸：600x400像素
    plt.rcParams['image.interpolation'] = 'nearest'  # 图像绘制插值方法，nearest在小图像放大操作中性能良好，这里也可以用None、none
    plt.rcParams['image.cmap'] = 'gray'  # 设置图像绘制颜色为灰度
    fig, axarr = plt.subplots(1, 2)
    axarr[0].set_title('x')
    axarr[0].imshow(x[0, :, :, 0])
    axarr[1].set_title('x_pad')
    axarr[1].imshow(x_pad[0, :, :, 0])
    plt.show()


# 运行测试
if __name__ == "__main__":
    test()
