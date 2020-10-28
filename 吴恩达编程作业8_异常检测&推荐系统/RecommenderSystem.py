# 推荐系统 --- 基于协同过滤算法的电影推荐
# Feb.28th

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as scio

# 导入数据
# Params = scio.loadmat('ex8_movieParams.mat')
# # X, theta = Params['X'], Params['Theta']
# # num_users, num_movies, num_features = Params['num_users'], Params['num_movies'], Params['num_features']
movies = scio.loadmat('ex8_movies.mat')
Y0, R0 = movies['Y'], movies['R']  # 训练集

# 加上一列作为新用户的偏好数据
my_ratings = np.zeros([Y0.shape[0], 1])
my_ratings[1] = 4
my_ratings[98] = 2
my_ratings[7] = 3
my_ratings[12] = 5
my_ratings[54] = 4
my_ratings[64] = 5
my_ratings[66] = 3
my_ratings[69] = 5
my_ratings[183] = 4
my_ratings[226] = 5
my_ratings[355] = 5
my_R = np.array([1 if my_ratings[i] != 0 else 0 for i in range(Y0.shape[0])])
Y = np.column_stack((Y0, my_ratings))
R = np.column_stack((R0, my_R))

# 参数初始化
num_movies, num_users = Y.shape  # 电影数，用户数
num_features = 100  # 特征数
X = np.random.uniform(-0.5, 0.5, [num_movies, num_features])  # 用户的参数
theta = np.random.uniform(-0.5, 0.5, [num_users, num_features])  # 电影的参数

# 训练
epochs = 300  # 迭代次数
alpha = 1  # 学习率
lamda = 0  # 正则化参数
cost = []
for epoch in range(epochs):
    # 均方误差损失函数
    J = 1/2 * np.mean(pow(np.multiply(np.matmul(X, theta.T), R) - Y, 2)) \
        + lamda/2*np.sum(pow(theta, 2)) + lamda/2*np.sum(pow(X, 2))
    cost.append(J)
    print('Epoch:%3d    Cost:%.5f' % (epoch, J))
    # 计算梯度
    delta_X = np.matmul(np.multiply(np.matmul(X, theta.T), R) - Y, theta) + lamda*X
    delta_theta = np.matmul((np.multiply(np.matmul(X, theta.T), R) - Y).T, X) + lamda*theta
    # 参数更新
    X = X - alpha * delta_X / num_users
    theta = theta - alpha * delta_theta / num_movies
print('训练完成！')
plt.plot(range(len(cost)), cost)
plt.show()

# 给出新用户前10部推荐电影
new_user_movies = np.matmul(X, theta[943, :].T)  # 为第944个用户计算每部电影的得分
index = 1 + np.argsort(-new_user_movies)  # 按照得分从大到小排序
print('前10部推荐电影序号：', index[:10])












