# SVM
# Feb.28th
# 理解了：参数C、gamma的意义

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
import cap
from sklearn import svm

# 导入数据
data = scio.loadmat('ex6data2.mat')['X']
label = scio.loadmat('ex6data2.mat')['y'].reshape(-1)

# SVM
clf = svm.SVC(kernel='rbf', C=1, gamma=50)
clf.fit(data, label)

# 画出决策边界
xx = np.arange(np.min(data[:, 0]), np.max(data[:, 0]), 0.01)
yy = np.arange(np.min(data[:, 1]), np.max(data[:, 1]), 0.01)
xx, yy = np.meshgrid(xx, yy)
zz = clf.predict(np.column_stack((xx.reshape(-1), yy.reshape(-1))))
zz = zz.reshape(xx.shape)
plt.contour(xx, yy, zz)
cap.plot_original_data(np.column_stack((data, label)))  # 原始样本
plt.show()

















