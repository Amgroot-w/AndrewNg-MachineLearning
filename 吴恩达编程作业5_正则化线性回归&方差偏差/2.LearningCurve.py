# 绘制学习曲线
# Feb.24th

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
import cap

# 导入数据
train_x0 = cap.normalize(scio.loadmat('ex5data1.mat')['X'])  # 归一化
train_y0 = scio.loadmat('ex5data1.mat')['y']
validation_x = cap.normalize(scio.loadmat('ex5data1.mat')['Xval'])  # 归一化
validation_y = scio.loadmat('ex5data1.mat')['yval']
test_x = cap.normalize(scio.loadmat('ex5data1.mat')['Xtest'])  # 归一化
test_y = scio.loadmat('ex5data1.mat')['ytest']

m = train_x0.shape[0]
train_cost = []
validation_cost = []
for i in range(1, m+1):
    # 线性拟合
    train_x = train_x0[:i, :]
    train_y = train_y0[:i, :]
    epochs = 50
    alpha = 1
    lamda = 0
    w = np.random.uniform(-0.5, 0.5)
    b = 0
    cost = []
    for epoch in range(epochs):
        h = w * train_x + b
        J = 1/2 * np.mean(pow(h-train_y, 2))
        cost.append(J)
        dw = np.mean(np.multiply(h-train_y, train_x)) + 1/m*lamda*w
        db = np.mean(h-train_y)
        w = w - alpha * dw
        b = b - alpha * db
    train_cost.append(J)
    J1 = 1/2*np.mean(pow(w*validation_x+b-validation_y, 2))
    validation_cost.append(J1)
# 可视化
plt.plot(range(len(train_cost)), train_cost)
plt.plot(range(len(validation_cost)), validation_cost)
plt.show()

















