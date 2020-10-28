# 异常检测 --- 高维数据
# Feb.25th

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as scio
import math

# 计算概率---方法1.直接按列相乘
def compute_p1(x_, mu_, sigma_):
    a = 1 / np.sqrt(2 * math.pi * sigma_**2)
    b = np.exp(-(x_-mu_) ** 2 / (2 * sigma_**2))
    res0 = a * b  # 得到每个样本、每个特征的概率（m×n矩阵）
    res = np.prod(res0, axis=1).reshape(-1, 1)  # 直接按列相乘
    return res

# 计算概率---方法2.多元高斯分布
def compute_p2(x_, mu_, cov_):
    a = 1 / ((2 * math.pi) ** (n / 2) * np.linalg.det(cov_) ** 0.5)
    b = np.matmul(x_-mu_, np.linalg.inv(cov_))
    c = np.matmul(b, np.transpose(x_-mu))
    res0 = a * np.exp(-1/2 * c)  # 这一步得到的是m×m矩阵
    res1 = [res0[i, i] for i in range(res0.shape[0])]  # 取res0对角线元素，得到每个样本的概率
    res = np.array(res1).reshape(-1, 1)
    return res


# 导入数据
train_x = scio.loadmat('ex8data2.mat')['X']
validation_x = scio.loadmat('ex8data2.mat')['Xval']
validation_y = scio.loadmat('ex8data2.mat')['yval']

# 参数拟合
m, n = train_x.shape  # 样本数、特征数
mu = np.mean(train_x, 0)  # 均值
sigma = np.var(train_x, 0)  # 方差
cov = np.matmul((train_x-mu).T, train_x - mu) / m  # 协方差矩阵

# 计算每个样本的概率
train_p = compute_p2(train_x, mu, cov)  # 方法2.多元高斯分布

# 选择最优的阈值
validation_p = compute_p2(validation_x, mu, cov)  # 计算验证集样本的概率（方法2）
epsilons = np.arange(0.1*10**(-18), 10*10**(-18), 10**(-20))  # 阈值的取值范围

F1 = {'epsilon': [], 'F1': []}
for epsilon in epsilons:
    pred0 = [1 if validation_p[i] < epsilon else 0 for i in range(validation_p.shape[0])]
    pred = np.array(pred0).reshape(validation_y.shape)
    # 计算F1-Score
    TP = np.sum((validation_y == 1) & (pred == 1))
    FP = np.sum((validation_y == 0) & (pred == 1))
    FN = np.sum((validation_y == 1) & (pred == 0))
    P = TP / (TP + FP)
    R = TP / (TP + FN)
    F1['epsilon'].append(epsilon)
    F1['F1'].append(2 * P*R / (P+R))
# 可视化
plt.plot(F1['epsilon'], F1['F1'])
plt.xlabel('epsilon')
plt.ylabel('F1-Score')
plt.show()
F1 = pd.DataFrame(F1)
best_F1 = max(F1['F1'])  # 最优的F1-Score
best_epsilons = F1.loc[F1['F1'] == best_F1]['epsilon']  # 最优的阈值epsilon
best_epsilon = best_epsilons.iloc[0]  # epsilons中所有值都满足F1最大（取第一个值作为最优值）

# 根据选出的最优epsilon值，得到异常点的序列
train_pred0 = [1 if train_p[i] < best_epsilon
               else 0 for i in range(train_p.shape[0])]
train_pred = np.array(train_pred0).reshape([-1, 1])
index = []
for i in range(train_x.shape[0]):
    if train_pred[i] == 1:
        index.append(i)
anomalies = train_x[index]
print('异常点个数：', anomalies.shape[0])





