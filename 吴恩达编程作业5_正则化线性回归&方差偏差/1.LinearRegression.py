# 正则化线性回归
# Feb.24th
# 调试记录：
# 1. 调了2个小时，就一个变量，线性回归，怎么都不收敛。。。。。。
#       问题又出现在归一化上，这次想着是就单变量，应该不需要归一化，然而它死活都不收敛！
#    事实证明，拿到数据，先把他给归一化了再说！！！
# 2. 没办法重现吴恩达pdf里面的过拟合？
#
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
import cap

# 导入数据
train_x = cap.normalize(scio.loadmat('ex5data1.mat')['X'])  # 归一化
train_y = scio.loadmat('ex5data1.mat')['y']
validation_x = cap.normalize(scio.loadmat('ex5data1.mat')['Xval'])  # 归一化
validation_y = scio.loadmat('ex5data1.mat')['yval']
test_x = cap.normalize(scio.loadmat('ex5data1.mat')['Xtest'])  # 归一化
test_y = scio.loadmat('ex5data1.mat')['ytest']

# 线性拟合
epochs = 50
alpha = 1
lamda = 0
m = train_x.shape[0]
w = np.random.uniform(-0.5, 0.5)
b = 0
cost = []
for epoch in range(epochs):
    h = w * train_x + b
    J = 1/2 * np.mean(pow(h-train_y, 2))  # 均方误差损失函数
    cost.append(J)
    dw = np.mean(np.multiply(h-train_y, train_x)) + 1/m*lamda*w
    db = np.mean(h-train_y)
    w = w - alpha * dw
    b = b - alpha * db

# 可视化
plt.plot(range(len(cost)), cost)
plt.show()
plt.scatter(train_x, train_y)
plot_x = np.arange(0, 1, 0.01)
plot_y = w * plot_x + b
plt.plot(plot_x, plot_y)
plt.show()

















