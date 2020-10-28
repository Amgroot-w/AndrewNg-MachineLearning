# Logistic Regression -- 预测学生是否能被录取
# Feb.17th
# 调试记录：
# 1. 逻辑回归模型拟合本数据集，效果不太好，有时候根本分不开两类样本？？？？？
#       原因是没有归一化！！！！！之前是觉得两个特征都是得分，在同一个量度内（0~100分），
#    所以觉得不需要归一化，但是这样会引起一个问题，就是sigmoid函数的输出太剧烈。尝试过将
#    sigmoid函数换为线性激活函数（y=kx），但是效果并未改善，而且由于是逻辑回归，本来就
#    应该用sigmoid函数，所以问题的本质还是sigmoid的输入问题：如果不归一化，其输入范围
#    就不是（-1,1）了，导致大量的样本点的sigmoid输出要么几乎为1，要么几乎为0（对应cost
#    的值要么是inf，要么是nan），而归一化后的数据，都分布在（-1,1）内，sigmoid函数可以
#    将这个范围的数据分得很开，当然模型的训练效果就大大改善了。
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cap

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def cross_entropy(x, y):
    return np.mean((np.multiply(-y, np.log(x)) -
                          np.multiply((1-y), np.log(1-x))))

def predict(x, theta):
    x = np.column_stack((np.ones([x.shape[0]]), x))
    predx = np.matmul(x, theta)
    return np.array([1 if predx[i] > 0.5 else 0 for i in range(x.shape[0])])

# 绘制原始样本
def plot_original_data():
    colors = ['c', 'orange']
    marker = ['o', 's']
    for i in range(2):
        score1 = data.loc[data['decision'] == i]['score1']
        score2 = data.loc[data['decision'] == i]['score2']
        plt.scatter(score1, score2, c=colors[i], marker=marker[i], s=50, linewidths=0.8, edgecolors='k')

# 绘制决策边界
def plot_decision_boundary(X, theta):
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.05
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = predict(np.c_[xx.ravel(), yy.ravel()], theta)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)  # 决策边界
    plot_original_data()  # 原始样本分布图
    plt.show()


# 导入数据
data = pd.read_csv('ex2data1.csv', names=['score1', 'score2', 'decision'])

# 初始化
m = data.shape[0]
# 归一化 -- 很重要的一步！！！！！
data.iloc[:, :2] = (data.iloc[:, :2] - np.mean(data.iloc[:, :2])) / np.std(data.iloc[:, :2])
x = np.column_stack((np.ones([m]), data.iloc[:, :2]))  # 加上第一列截距项
y = np.array([data['decision']]).T
theta = np.random.uniform(low=-5, high=5, size=[3, 1])
cost_history = {'epoch': [], 'cost': []}

# 训练
epochs = 10000
learning_rate = 0.01
print('************ 开始训练 ************')
for epoch in range(epochs):
    h = sigmoid(np.matmul(x, theta))       # 假设函数 h(θ)
    J = cross_entropy(h, y)                # 交叉熵损失
    delta = 1/m * np.matmul(x.T, h-y)      # 计算梯度
    theta = theta - learning_rate * delta  # 更新参数θ
    cost_history['epoch'].append(epoch)    # 记录误差cost
    cost_history['cost'].append(J)
    print('Epoch:%d, Cost:%.4f' % (epoch, J))
print('************ 训练完成 ************')
plt.plot(cost_history['epoch'], cost_history['cost'])
plt.show()

# 可视化决策边界方法1
plotx = range(-2, 3, 1)
ploty = -theta[1]/theta[2]*plotx + (0.5-theta[0])/theta[2]
plt.plot(plotx, ploty, '-r')
plot_original_data()  # 样本分布图（归一化后）
plt.show()

# # 可视化决策边界方法2
# plot_decision_boundary(data.values, theta)

# 调用cap.py
# a = cap.feature_mapping(data.values[:, 0], data.values[:, 1], degree=1)
cap.plot_decision_boundary(data.values, theta)


