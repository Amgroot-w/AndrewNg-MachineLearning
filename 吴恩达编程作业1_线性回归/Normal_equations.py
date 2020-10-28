# 线性回归--标准方程解法
# Feb.16th
# 用[1650, 3]的输入检验，发现与线性回归的结果相差不大，说明线性回归模型拟合的比较好
import numpy as np
import pandas as pd

data = pd.read_csv('ex1data2.csv', header=None)
x = data.iloc[:, :2].values
y = data.iloc[:, 2].values

a = np.linalg.inv(np.matmul(np.transpose(x), x))
b = np.matmul(a, np.transpose(x))
theta = np.matmul(b, y)

test = np.array([1650, 3])
pred = np.matmul(test, theta.T)
print('1650-square-foot house with 3 bedrooms, its price is about: %.2f$' % pred)








