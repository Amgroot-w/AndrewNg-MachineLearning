# k-means 聚类算法（已集成到cap.py）
# Feb.12th

import numpy as np

# K-means算法
# 输入：data矩阵（m×n）：m个样本，n个特征；K：想要聚为多少类；iteration：迭代次数
# 输出：c（m×1）为所有样本的聚类结果（范围：0~K-1），centroid（K×n）：K个聚类中心的坐标
def kmeans(data, K, iteration):
    # 求a, b的欧式距离
    def distance(a, b):
        return pow((sum(pow(a - b, 2))), 0.5)
    # 随机选择初始聚类中心
    m = data.shape[0]  # 样本数m
    index = np.random.random_integers(1, m, K)  # 在1~m中产生K个随机数
    centroid = data[index]  # 初始聚类中心
    d = np.zeros((m, K))  # 距离矩阵
    c = np.zeros(m)  # 类别向量(0,1,...,K)
    for iter in range(iteration):
        print('Iteration:', iter)
        # 簇分配
        c_last = c  # 保存上一次迭代的分类结果
        for i in range(m):
            for k in range(K):
                # 计算第i个样本距离第k个聚类中心的距离
                d[i, k] = distance(data[i], centroid[k])
        c = d.argmin(axis=1)  # 按行取最小值，得到每个样本所属类别
        # 终止条件：没有样本被重新分类
        if (c == c_last).all():
            break
        # 移动聚类中心
        for k in range(K):
            # 找到所有属于第k类的样本
            examples_belong_to_k = data[[i for i, x in enumerate(c) if x == k]]
            centroid[k] = np.mean(examples_belong_to_k, 0)

    return c, centroid








