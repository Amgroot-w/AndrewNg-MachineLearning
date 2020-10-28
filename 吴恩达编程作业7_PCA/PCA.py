# PCA算法（已集成到cap.py）
# Feb.13th
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio

# 均值归一化（去中心化）
def normalize(x):
    x_mean = np.mean(x, axis=0)
    # x_range = np.max(x, axis=0) - np.min(x, axis=0)
    return x-x_mean, x_mean

# 选择合适的K
def find_k(s):
    # s表示对角阵，但是s是1×n矩阵，只列出了对角线元素
    k = s.shape[0]
    sum_s = np.sum(s)
    sum_k = sum_s
    while (sum_k / sum_s) >= 0.99:
        sum_k = sum_k - s[k-1]
        k = k-1
    return k+1

# 可视化图片压缩效果
def display(images, pixel, size):
    # images为图片集数据矩阵
    # pixel表示像素 (e.g. [32×32]）
    # size表示可视化的行列数（e.g. [10×10]）
    m = images.shape[0]  # 图片总数
    for i in range(m):
        image = images[i, :].reshape([pixel[0], pixel[1]])
        plt.subplot(size[0], size[1], i+1)
        plt.imshow(image.T)
        plt.axis('off')
        # plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
        #                     wspace=0, hspace=0)
    plt.show()

# PCA算法
# 输入：归一化后的训练集
# 输出：u为n×n特征矩阵，k为特征数目，满足“99% of variance is retained”
def pca(data):
    # sigma = np.matmul(data.T, data) / data.shape[0]  # 协方差矩阵
    # 上面一步求协方差和下面的svd函数中求协方差重了！（详见PCA_faces.py调试记录）
    [u, s, v] = np.linalg.svd(data)  # 调用svd函数
    # 注：返回的s并不是一个n×n矩阵，而是1×n的元组，表示对角线元素！
    k = find_k(s)  # 满足“99%的方差均被保留”的最小的k
    return [u, v, k]













