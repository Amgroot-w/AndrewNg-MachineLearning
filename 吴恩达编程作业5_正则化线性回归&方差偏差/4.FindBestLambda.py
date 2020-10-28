# 通过验证集找到最优的λ
# Feb.24th

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
import cap

# 导入数据
train_x0 = scio.loadmat('ex5data1.mat')['X']
train_y = scio.loadmat('ex5data1.mat')['y']
validation_x0 = scio.loadmat('ex5data1.mat')['Xval']
validation_y = scio.loadmat('ex5data1.mat')['yval']

# 一维 -> n维
degree = 8
train_x = cap.polyfeatures(train_x0, degree)
train_x = cap.normalize(train_x, 'zscore')
train_x[:, 0] = 1

validation_x = cap.polyfeatures(validation_x0, degree)
validation_x = cap.normalize(validation_x, 'zscore')
validation_x[:, 0] = 1

m = train_x.shape[0]
train_cost = []
validation_cost = []
lamdas = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])

for lamda in lamdas:
    # 线性拟合
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
    train_cost.append(J)
    J1 = 1/2 * np.mean(pow(np.matmul(validation_x, theta)-validation_y, 2))
    validation_cost.append(J1)

# 学习曲线
plt.plot(lamdas, train_cost, 'b')
plt.plot(lamdas, validation_cost, 'g')
plt.show()











