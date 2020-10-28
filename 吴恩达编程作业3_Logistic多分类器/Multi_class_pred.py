# 多分类逻辑回归--手写数字识别
# Feb.18th
# 运行本代码得到 准确率ac

import pandas as pd
import numpy as np
from scipy.io import loadmat

images = loadmat('ex3data1.mat')['X']
labels = loadmat('ex3data1.mat')['y']
x = np.column_stack((np.ones([images.shape[0], 1]), images))  # 加上第一列
thetas = pd.read_csv('thetas.csv', header=None).values

pred = np.ones(labels.shape[0]) + np.argmax(np.matmul(x, thetas), 1)  # 预测类别
pred = np.array(pred).reshape(labels.shape)  # 规范格式
ac = np.mean(np.equal(pred, labels))  # 计算准确率

print("训练集准确率:", 100*ac, '%')



