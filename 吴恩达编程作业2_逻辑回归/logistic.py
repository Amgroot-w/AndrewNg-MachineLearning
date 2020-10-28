# logistic算法（已集成在cap.py中）
# 二分类

import numpy as np
import matplotlib.pyplot as plt
import cap

# Logistic回归
# 输入：特征x：m×(1+n)
#      标签y：m×1，标签取值为0和1，二分类
#      迭代次数epochs
#      学习率alpha
#      正则化参数lamda
# 输出：参数theta((1+n)×1)
# ** Note: 输入矩阵x必须满足：①已经归一化；②第一列为1（截距项）
#          输出参数theta的第一个数为theta0（截距项系数）
def logistic(x, y, epochs, alpha, lamda):
    m = x.shape[0]  # 样本数
    n = x.shape[1]  # 特征数
    theta = np.random.uniform(-1, 1, [x.shape[1], 1])  # 参数初始化
    delta = np.zeros([n, 1])  # 梯度初始化
    cost_history = {'epoch': [], 'cost': []}  # 字典记录误差变化
    # 训练
    # print('************ 开始训练 ************')
    for epoch in range(epochs):
        # 假设函数h(θ)
        h = cap.sigmoid(np.matmul(x, theta))
        # 交叉熵损失 + 正则化项
        J = cap.cross_entropy(h, y) + lamda * 1/(2*m) * np.sum(pow(theta[1:n, :], 2))
        # 计算梯度
        delta[0, :] = 1/m * np.matmul(x.T[0, :], h-y)  # theta0不加正则化
        delta[1:n, :] = 1/m * np.matmul(x.T[1:n, :], h-y) + lamda*1/m*theta[1:n, :]
        # 参数更新
        theta = theta - alpha * delta
        # 记录误差cost
        cost_history['epoch'].append(epoch)
        cost_history['cost'].append(J)
        # print('Epoch:%d, Cost:%.4f' % (epoch, J))
    # print('************ 训练完成 ************')
    # 可视化误差曲线
    plt.plot(cost_history['epoch'], cost_history['cost'])
    plt.show()

    return theta
