# 线性回归--单变量
# Feb.16th
# 调试记录：
# 1. 如果不归一化，模型为什么会发散的，无法收敛？？？？？
#       其实不是，单变量不需要归一化模型也可以收敛，而之前发散的原因是：把公式写错了：θ的更新
#    少乘了1/m（主要原因），J(θ)少乘了1/2.
#    Feb.24th记录：特么的还是得归一化，单变量不归一化死活都不都收敛。。。。。。
# 2. 训练模型得到了适用于归一化后数据的参数theta，但是给定一个原始scale的x，怎么得到预测值？
#       给定测试值，将其按照训练集的数据（均值、方差）归一化，然后带入模型得到预测值。
# 3. 归一化是只对X，不对Y？
#       归一化的目的是解决各个特征的量度不统一问题，有的特征值整体很大，有的整体很小，这很容
#    易造成某一个特征对模型的训练器决定作用而忽视了其他特征。因此要对特征进行归一化，而不是Y
#    值（这样也能保证将测试值带入模型后，得到的预测结果就是实际的Y值量度）。
# 4.Feb.24th记录：本程序对theta0也添加了正则化（实际上不用）

import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 导入数据
data0 = pd.read_csv('ex1data1.csv', header=None)  # 读取cvs文件
# data = (data0 - np.min(data0)) / (np.max(data0) - np.min(data0))  # 归一化
# data = (data0 - np.mean(data0)) / np.std(data0)  # 归一化
data = data0  # 不进行归一化
train_x = pd.DataFrame(np.column_stack((np.ones(data.shape[0]),
                                        data.iloc[:, 0].values)))  # 加入第一列（全1）
train_y = pd.DataFrame(data.iloc[:, 1])

# 搭建TensorFlow模型
# 参数
m = train_x.shape[0]  # 样本个数
n = train_x.shape[1]  # 变量个数
learning_rate = 0.01  # 学习率
epochs = 1000  # 迭代次数
# 绘图
x = tf.placeholder(tf.float32, [None, n])
y = tf.placeholder(tf.float32, [None, 1])
theta = tf.Variable(tf.random_normal([n, 1]))
pred = tf.matmul(x, theta)
J = 1/2 * tf.reduce_mean(tf.pow(pred-y, 2))
theta_update = 1/m * learning_rate * tf.matmul(tf.transpose(x), pred-y)
theta = tf.assign_add(theta, -theta_update)

# 启动会话
with tf.Session() as sess:
    cost_history = {'epoch': [], 'cost': []}
    sess.run(tf.global_variables_initializer())

    print('********** 开始训练 **********')
    for epoch in range(epochs):
        t, cost = sess.run([theta, J], feed_dict={x: train_x, y: train_y})  # 喂数据
        cost_history['epoch'].append(epoch)  # 保存每次迭代的cost数据
        cost_history['cost'].append(cost)
        print('epoch:%3d   cost:%4f' % (epoch, cost))
    print('********** 训练完成 **********')

def pred(x):
    return t[0] + x*t[1]


# 可视化
plt.plot(cost_history['epoch'], cost_history['cost'])  # cost曲线
plt.show()
plt.plot(data.iloc[:, 0], data.iloc[:, 1], '.')  # 原始样本
plt.plot([5, 25], [pred(5), pred(25)], '--r')
plt.show()

theta0 = np.arange(-10, 100, 10)
theta1 = np.arange(-1, 100, 4)
J_theta = np.zeros([theta0.shape[0], theta1.shape[0]])
for i in range(theta0.shape[0]):
    for j in range(theta1.shape[0]):
        J_theta[i][j] = 1 / 2 * np.mean(pow(np.matmul(
            train_x, np.array([[theta0[i]], [theta1[j]]])) - train_y, 2))
ax = Axes3D(plt.figure())
theta0, theta1 = np.meshgrid(theta0, theta1)
ax.plot_surface(theta0, theta1, J_theta.T)
plt.show()
plt.contour(theta0, theta1, J_theta.T)
plt.show()





