# 用最优的λ，计算测试集误差
# Feb.24th

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
import cap

# 导入数据
train_x0 = scio.loadmat('ex5data1.mat')['X']
train_y = scio.loadmat('ex5data1.mat')['y']

# 一维 -> n维
degree = 8
train_x = cap.polyfeatures(train_x0, degree)
train_x = cap.normalize(train_x, 'zscore')
train_x[:, 0] = 1

m = train_x.shape[0]
lamda = 3
epochs = 1000
alpha = 0.03
theta = np.random.uniform(-1, 1, [degree+1, 1])
delta = np.zeros(theta.shape)
cost = []
for epoch in range(epochs):
    h = np.matmul(train_x, theta)
    J = 1/2 * np.mean(pow(h-train_y, 2))  # 均方误差损失函数
    cost.append(J)
    delta = 1/m * np.matmul(train_x.T, h-train_y) + 1/m*lamda*theta
    delta[0, :] = 1/m * np.matmul(train_x[:, 0].T, h - train_y) + 1/m*lamda*theta[0, :]
    theta = theta - alpha * delta

# 计算测试集误差
test_x0 = scio.loadmat('ex5data1.mat')['Xtest']
test_y = scio.loadmat('ex5data1.mat')['ytest']
test_x = cap.polyfeatures(test_x0, degree)
test_x = cap.normalize(test_x, 'zscore')
test_x[:, 0] = 1
J2 = 1 / 2 * np.mean(pow(np.matmul(test_x, theta) - test_y, 2))
print('测试集误差为：%.2f' % J2)


