# SVM
# Feb.28th

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
import cap
from sklearn import svm

# 导入数据
data = scio.loadmat('ex6data1.mat')['X']
label = scio.loadmat('ex6data1.mat')['y'].reshape(-1)

# SVM
clf = svm.SVC(kernel='linear', C=1)
clf.fit(data, label)

# 画出决策边界
a = - clf.coef_[0, 0] / clf.coef_[0, 1]  # 斜率
b = - clf.intercept_[0] / clf.coef_[0, 1]  # 截距
xx = np.arange(0, 5, 0.1)
yy = a * xx + b
plt.plot(xx, yy)  # 决策边界
cap.plot_original_data(np.column_stack((data, label)))  # 原始样本
plt.show()

















