# SVM
# Feb.28th

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
import cap
from sklearn import svm

# 导入数据
data = scio.loadmat('ex6data3.mat')['X']
label = scio.loadmat('ex6data3.mat')['y'].reshape(-1)
data_val = scio.loadmat('ex6data3.mat')['Xval']
label_val = scio.loadmat('ex6data3.mat')['yval'].reshape(-1)

# 网格搜索：找到最优的超参数C、gamma
para1 = np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 50, 70, 100])
para2 = np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 50, 70, 100])
ac = np.zeros([para1.shape[0], para2.shape[0]])
max_index = [0, 0]
for i in range(para1.shape[0]):
    for j in range(para2.shape[0]):
        clf = svm.SVC(kernel='rbf', C=para1[i], gamma=para2[j])
        clf.fit(data, label)
        pred_val = clf.predict(data_val)
        ac[i, j] = np.mean(pred_val == label_val)
        if ac[i, j] > ac[max_index[0], max_index[1]]:
            max_index = [i, j]

# SVM
clf = svm.SVC(kernel='rbf', C=para1[max_index[0]], gamma=para2[max_index[1]])
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

















