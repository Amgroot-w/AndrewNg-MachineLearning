# Logistic Regression with Regularization -- 预测芯片能否通过质量保证
# Feb.17th
# 调试记录：
# 1. 不要随便点黄色警告框！！！！！！！！！！
#       调试时，matplotlib.pyplot.contourf的参数”cmap=plt.cm.Spectral“
#    会有警告，点了下“Creat class 'Spectral'”，然后这个函数就用不了。。。
# 2. 正则化参数λ：
#       1）没有归一化的情况下：degree设为10。λ为0时，会发生过拟合；λ增大至1时，
#    拟合比较好；λ为4时，已经能观察到明显的欠拟合，继续增至λ=6，处于收敛的边缘，
#    大于6时，模型发散。
#       2）归一化处理后：degree设置为8不变。λ=0，能观察到明显的过拟合；增大至λ
#    =1，能观察到欠拟合情况缓解了，但是仍存在；反复调试发现最优的λ值在10~70之间，
#    此时模型拟合效果最好（人为观察）；继续增大，发现很难模型欠拟合，直到λ=100，
#    模型拟合效果都还不错；继续增到λ=300，可以观察到欠拟合，继续增大至3000左右，
#    模型发散。
# 3. 关于是否归一化：
#       由于上个代码出现的问题，现在考虑是否在本例程添加归一化：刚开始不加归一化，
#    发现分类效果、可视化都很正常，只有一点就是不能很明显的观察到正则化参数的作用
#    （即不管调大degree还是去掉正则化，模型都好像不会过拟合），估计原因是：样本
#    的数值本来就大部分分布在（-1,1）之间，只有少数几个样本在此区间外。当添加了归
#    一化处理后，所有都正常了：在degree=6的时候，去掉正则化项即可观察到明显的过
#    拟合，然后稍微添加正则化参数（λ=1）即可消除过拟合现象。所以结论是：以后不管
#    特征是怎么样的，先进行归一化处理再说！
# 4. Feb.18th, 对本代码进行封装 -- logistic.py
# 5. Feb.18th, 调试多分类logistic算法时，发现始终无法收敛，将初始化参数范围由
#    （-5,5）改为（-1,1）时，模型收敛。说明参数的初始化方法的选择非常重要！！！

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from logistic import logistic
import cap

# 二维特征扩展为高维多项式特征(已集成到cap.py)
def feature_mapping(x1, x2, degree):
    res = np.ones([x1.shape[0], 1])
    for i in range(1, degree+1):  # i:1~6
        for j in range(i+1):      # j:0~i
            new_feature = np.multiply(x1**(i-j), x2**j)
            res = np.column_stack((res, new_feature))
    return res

# 预测类别
def predict(x, theta):
    predx = np.matmul(x, theta)
    return np.array([1 if predx[i] > 0.5 else 0 for i in range(x.shape[0])])

# 绘制原始样本
def plot_original_data():
    colors = ['c', 'orange']
    marker = ['o', 's']
    for i in range(2):
        score1 = data.loc[data['result'] == i]['score1']
        score2 = data.loc[data['result'] == i]['score2']
        plt.scatter(score1, score2, c=colors[i], marker=marker[i], s=50, linewidths=0.8, edgecolors='k')

# 绘制决策边界
def plot_decision_boundary(X, theta):
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.05
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = predict(cap.feature_mapping(xx.ravel(), yy.ravel(), degree=8), theta)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)  # 决策边界
    plot_original_data()  # 原始样本分布图
    plt.show()


# 导入数据
data = pd.read_csv('ex2data2.csv', names=['score1', 'score2', 'result'])
data.iloc[:, :2] = (data.iloc[:, :2] - np.mean(data.iloc[:, :2])) / np.std(data.iloc[:, :2])  # 归一化
x = cap.feature_mapping(data['score1'], data['score2'], degree=8)
y = np.array([data['result']]).T

# Logistic回归
theta = cap.logistic(x, y, epochs=10000, alpha=0.05, lamda=20)

# 可视化决策边界
plot_decision_boundary(data.values, theta)



