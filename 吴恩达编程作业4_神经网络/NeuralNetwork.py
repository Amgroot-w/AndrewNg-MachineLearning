# 神经网络 -- 手写数字识别
# Feb.19th
# 要编写的子函数：权重按照公式随机初始化（page7）
# 调试记录：
# 1. 怎么又不能收敛？？？？？
#       观察变量值发现，隐层输入太大，导致sigmoid函数的输出几乎全是1，引起这个的原因是权重的初
#    始值设置的太高，重新设置为（-0.5,0.5）之后就解决了这个小问题，但是模型照样发散，说明不是这
#    个问题引起的（实际上，把下面的问题解决后，在把参数初始值调高到（-5,5），模型也能收敛）。
#       问题其实出现在梯度的更新上。算完误差的反向传播后，输出层误差和隐层误差均为矩阵形式表示，
#    即没有进行求和，表示的意义是“某个样本在某个节点上的误差”；而在接下来计算梯度的时候，是将上
#    面两个误差按照样本数目的维度给降维了，即它描述的对象不再是”某个样本“，而是”某个节点“！！！
#    因此，必须在处理的时候，前面除以样本个数（1/m），这样每一次迭代的梯度值就在正常范围内了。
# 2. cost曲线：先降低、再升高？？？？？（U型）
#       这是因为输出层误差的公式写错了，写成了交叉熵的形式。注意：对于输出层误差来说，就是h-y就
#    行了，而交叉熵的形式是用来算损失函数的，这一点搞混了！而且注意另一点：损失函数的返回值永远是
#    一个数，而不是矩阵形式！！！
# 3. 为什么加上lamda却收敛了？？？？？为什么再次尝试去掉lamda，竟然又可以了？？？？？
#       没弄清楚。。。
# 4. 训练集准确率95%左右，测试集准确率10%左右？？？？？
#       两个差距太大了，应该不是过拟合的原因，猜测是测试数据出了问题。果然，将lamda调到很大，测
#    试准确率照样低，说明并不是过拟合。仔细检查发现，在test_x归一化后，我调出他的最值，发现最大
#    值偶尔会是inf，最小值一般是-1左右（即最值不在标准的（0,1）之间），这是因为处理测试集的归一
#    化时，用的是训练集的max、min（用zscore方法归一化时也是同样的问题），于是出现了这个情况，但
#    是即使有时候测试集的max和min都是正常的数，最终结果也是准确率极低。至于原因，猜测是因为这样处
#    理会改变测试集原有的信息，于是输入训练好的模型，模型就“不认识”它了（用自己的max和min归一化
#    就不会改变原有信息，可以通过可视化观察到）。所以以后测试集、验证集都按照标准的归一化方法来做吧。
# 5. 上个问题，用训练集的max，min来归一化测试集，为什么可视化后，显示测试集并没有改变什么信息？？？
#       未解决

import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
import cap

# 导入数据
images = loadmat('ex4data1.mat')['X']  # 图片
labels = loadmat('ex4data1.mat')['y']  # 标签
m = images.shape[0]  # 样本总数
n = images.shape[1]  # 特征数

# 打乱数据
data = np.column_stack((images, labels))
np.random.shuffle(data)  # 洗牌
images = data[:, :n]
labels = data[:, n].reshape([m, 1])

# 分配训练集、测试集
train_num = int(m * 0.7)  # 训练样本数
test_num = int(m * 0.3)   # 测试样本数
train_x = images[:train_num, :]
train_y = labels[:train_num, :]
test_x = images[train_num:m, :]
test_y = labels[train_num:m, :]

# 归一化处理
train_x = cap.normalize(train_x, 'maxmin')  # 特征归一化处理
# test_x = (test_x - train_max) / (train_max - train_min)  # 测试集用训练集的max、min归一化(不可行)
test_x = cap.normalize(test_x, 'maxmin')

# 异常值处理
# 采用MaxMin归一化，若某列特征的max=min(=0)，则返回值为nan；采用Zscore归一化，若某列特征全相等，则方差为0，返回值为nan.
train_x = pd.DataFrame(train_x).fillna(0).values  # nan替换为0
test_x = pd.DataFrame(test_x).fillna(0).values  # nan替换为0

# 标签采用one-hot编码
train_y = cap.onehot(train_y - 1)
test_y = cap.onehot(test_y - 1)

# 搭建神经网络框架
input_num = 400  # 输入节点数
hidden_num = 25  # 隐层节点数
output_num = 10  # 输出节点数

# 初始化权重
# w1 = np.random.uniform(-1, 1, [input_num, hidden_num])  # 普通初始化方法
# w2 = np.random.uniform(-1, 1, [hidden_num, output_num])
threshold = np.sqrt(6) / np.sqrt(400+25)
w1 = np.random.uniform(-threshold, threshold, [input_num, hidden_num])  # 公式初始化方法
threshold = np.sqrt(6) / np.sqrt(25+10)
w2 = np.random.uniform(-threshold, threshold, [hidden_num, output_num])
b1 = np.zeros(hidden_num)
b2 = np.zeros(output_num)

# 设置超参数
alpha = 1  # 学习率
lamda = 0  # 正则化参数
epochs = 2000  # 迭代次数
# 训练
cost = []
for epoch in range(epochs):
    # 前向传播
    hidden_in = np.matmul(train_x, w1) + b1
    hidden_out = cap.sigmoid(hidden_in)
    network_in = np.matmul(hidden_out, w2) + b2
    network_out = cap.softmax(network_in)
    # 记录总误差
    J = cap.cross_entropy(network_out, train_y) \
        + 1/(2*m) * lamda * (np.sum(w2**2)+np.sum(w1**2))
    cost.append(J)
    # 反向传播
    output_delta = network_out - train_y
    hidden_delta = np.multiply(np.matmul(output_delta, w2.T), np.multiply(hidden_out, 1-hidden_out))
    # 梯度更新
    dw2 = 1/train_num * (np.matmul(hidden_out.T, output_delta) + lamda*w2)
    db2 = 1/train_num * np.matmul(np.ones([train_num, 1]).T, output_delta)
    dw1 = 1/train_num * (np.matmul(train_x.T, hidden_delta) + lamda*w1)
    db1 = 1/train_num * np.matmul(np.ones([train_num, 1]).T, hidden_delta)
    w2 = w2 - alpha*dw2
    w1 = w1 - alpha*dw1
    b2 = b2 - alpha*db2
    b1 = b1 - alpha*db1
    # 展示训练过程
    if epoch % 20 == 0:
        print('Epoch:%4d     cost:%.4f' % (epoch, J))
# 可视化cost曲线
plt.plot(range(epochs), cost)
plt.show()

# 训练集准确率
pred = np.argmax(network_out, axis=1)
train_y = np.argmax(train_y, axis=1)
ac = np.mean(np.equal(pred, train_y))
print("训练集准确率：", "%.2f" % (ac*100), '%')

# 测试集准确率
hidden_in = np.matmul(test_x, w1) + b1
hidden_out = cap.sigmoid(hidden_in)
network_in = np.matmul(hidden_out, w2) + b2
network_out = cap.sigmoid(network_in)
test_pred = np.argmax(network_out, axis=1)
test_y = np.argmax(test_y, axis=1)
test_ac = np.mean(np.equal(test_pred, test_y))
print("测试集准确率：", "%.2f" % (test_ac*100), '%')

# 可视化隐层节点权重
cap.display(w1.T, [20, 20], [5, 5])

