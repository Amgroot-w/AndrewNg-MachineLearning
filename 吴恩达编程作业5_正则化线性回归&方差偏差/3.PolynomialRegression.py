# 多项式回归
# Feb.24th

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
import cap

# 导入数据
train_x00 = scio.loadmat('ex5data1.mat')['X']
train_y0 = scio.loadmat('ex5data1.mat')['y']
validation_x0 = cap.normalize(scio.loadmat('ex5data1.mat')['Xval'])  # 归一化
validation_y = scio.loadmat('ex5data1.mat')['yval']

# 一维 -> n维
degree = 8
train_x0 = cap.polyfeatures(train_x00, degree)
train_x0 = cap.normalize(train_x0, 'zscore')
train_x0[:, 0] = 1

validation_x = cap.polyfeatures(validation_x0, degree)
validation_x = cap.normalize(validation_x)
validation_x[:, 0] = 1

m = train_x0.shape[0]
train_cost = []
validation_cost = []
for i in range(1, m+1):
    train_x = train_x0[:i, :]
    train_y = train_y0[:i, :]
    # 线性拟合
    epochs = 1000
    alpha = 0.03
    lamda = 1
    theta = np.random.uniform(-1, 1, [degree+1, 1])
    delta = np.zeros(theta.shape)
    cost = []
    for epoch in range(epochs):
        h = np.matmul(train_x, theta)
        J = 1/2 * np.mean(pow(h-train_y, 2))  # 均方误差损失函数
        cost.append(J)
        delta = 1/i * np.matmul(train_x.T, h-train_y) + 1/i*lamda*theta
        delta[0, :] = 1/i * np.matmul(train_x[:, 0].T, h - train_y) + 1/i*lamda*theta[0, :]
        theta = theta - alpha * delta
    train_cost.append(J)
    J1 = 1/2 * np.mean(pow(np.matmul(validation_x, theta)-validation_y, 2))
    validation_cost.append(J1)

# 误差变化曲线
plt.plot(range(len(cost)), cost)
plt.show()

# 多项式拟合曲线
plot_x0 = np.arange(-40, 40, 0.01)
plot_x = cap.polyfeatures(plot_x0, degree)
plot_x = cap.normalize(plot_x, 'zscore')
plot_x[:, 0] = 1
plot_y = np.matmul(plot_x, theta)
plt.plot(plot_x0, plot_y)
plt.scatter(train_x00, train_y0)
plt.show()

# 学习曲线
plt.plot(range(len(train_cost)), train_cost)
plt.plot(range(len(validation_cost)), validation_cost)
plt.show()













