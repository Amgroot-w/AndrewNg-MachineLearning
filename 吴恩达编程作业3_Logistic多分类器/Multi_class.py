# 多分类逻辑回归--手写数字识别
# Feb.18th
"""
调试记录：
1. 模型始终无法收敛：将logistic.py的初始化方法改了，将参数的初始化范围从
   （-5,5）改为（-1,1），模型立即收敛。 --- 初始化方法很重要！
2. 迭代3000次，学习率=0.01，正则化参数lamda=2，准确率：76.88 %  **ac不是很理想
   迭代4000次，学习率=0.03，正则化参数lamda=5，准确率：88.18 %  **增大alpha和lamda起作用了
   迭代4000次，学习率=0.03，正则化参数lamda=10，准确率：88.50 %  **说明上一步不是lamda起的作用
   迭代4000次，学习率=0.3，正则化参数lamda=10，准确率：91.82 %  **确实是增大学习率的作用
   迭代4000次，学习率=0.3，正则化参数lamda=1，准确率：92.82 %  **减小正则化参数，ac反而升高
   迭代4000次，学习率=0.6，正则化参数lamda=0，准确率：93.74 %  **完全去掉正则化，开始提高alpha
   迭代3000次，学习率=1，正则化参数lamda=0，准确率：94.18 %  **增大alpha，同时减小多余的迭代次数
   迭代3000次，学习率=2，正则化参数lamda=0，准确率：94.78 %  **继续增大，cost曲线仍未出现波动
   迭代2000次，学习率=4，正则化参数lamda=0，准确率：95.02 %  **继续增大，cost曲线稍有波动
   迭代2000次，学习率=8，正则化参数lamda=0，准确率：90.34 %  **继续增大，cost曲线波动较大，且ac下降
   最优的超参数为：alpha=4，lamda=0，迭代次数：2000次（实际上200次左右就收敛了）
   选择以上最优超参数进行多次实验，准确率最高达到95.02%.
"""

import numpy as np
from scipy.io import loadmat
import csv
import cap

# 查找第k类、非第k类的标签
def find(list, target):
    target_index = []
    for i in range(list.shape[0]):
        if list[i] == target:
            target_index.append(i)
    return target_index


# 导入数据
images = loadmat('ex3data1.mat')['X']
labels = loadmat('ex3data1.mat')['y']

# 特征矩阵预处理
# x, x_max, x_min = normalize(images, 'maxmin')  # 归一化
# x = pd.DataFrame(x).fillna(0).values  # 剔除异常值
x = np.column_stack((np.ones([images.shape[0], 1]), images))  # 加上第一列

m = x.shape[0]  # 样本数
n = x.shape[1]  # 特征数

# 构建One-vs-all分类器
K = 10  # 类别数
thetas = np.zeros([n, K])
for k in range(1, K+1):
    y = np.zeros([m, 1])    # 训练标签初始化
    y[find(labels, k)] = 1  # 第k类样本标签记为1
    theta = cap.logistic(x, y, epochs=2000, alpha=4, lamda=0)  # 逻辑回归
    thetas[:, k-1] = theta[:, 0]
    print('第 %d 个logistic分类器训练完成' % k)

# 将训练结果写入csv文件
with open('thetas.csv', 'w') as datafile:
    writer = csv.writer(datafile, delimiter=',')
    writer.writerows(thetas)
print('训练结果已写入thetas.csv文件.')

# 接下来运行Multi_class_pred，即可得到在训练集的预测准确率




























