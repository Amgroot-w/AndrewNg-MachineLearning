# PCA算法实战--图片压缩（图片特征提取）
# Feb.13th
# PCA_faces.py调试记录：
# 1. 为什么重建输入能零误差？？？？？？为什么练习1不是？？？？
#       其实并不是零误差！以为是零误差的原因是把主成分数目k取得太大了，致使输出能够几乎完美
#   地重建输入，但其实还是有微小差异的，只不过肉眼无法分辨而已。把k值调小之后，例如调到99%对
#   应的k=22时，就能观察到微小差异，继续调小k，就会发现重建的输入与原图像差异越来越明显了。
# 2. 对图像压缩来说，用PCA压缩得到的特征U、V究竟有什么用？？？？
#       实验发现，PCA得到的特征只能用来重建训练过的图片，对于未经训练的图片（测试集），其重建
#   误差很大，而且可视化后能直观地观察出来。所以，PCA的目的不是为了得到一个模型，而是为了降维，
#   而降维的目的，有些是为了节省存储空间（如图片压缩处理），有些是为了进行提高算法效率（如利用
#   PCA进行数据预处理）。所以，特征矩阵U、V的作用是得到原始样本的压缩表示，而不是得到测试样本
#   的压缩表示。
# 3. 前面很多次调试时，实际上求了两次协方差矩阵！！！！！
#       svd函数内置的有求协方差矩阵的这一步骤，所以自己写的pca函数中，第一步的求协方差和svd函
#   数中的重了，相当于求了两次协方差，所以相当于对data的协方差进行了特征提取，而本应该对data进
#   行特征提取！所以造成了U和V矩阵一直不对劲，造成了U和V一直都是1024×1024，实际上应该是：U矩
#   阵为m×m，V矩阵为1024×1024。各自的作用是：U矩阵负责行降维（提取前k列为主要特征），V矩阵负
#   责列降维（提取前k行为主要特征）。本例中，之前一直用U矩阵降维是错的，应该用V矩阵降维才对。
#
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
from PCA import pca, normalize, display
import cap

# ************* 练习1：2D变1D *************
# 需要把PCA算法中的k设置为1，并添加可视化
data = scio.loadmat('ex7data1.mat')['X']
data, _ = normalize(data)  # 数据预处理--特征缩放
u, v, _ = pca(data)  # 执行PCA算法
k = 1
v_reduce = v[:k, :]
z = np.matmul(data, v_reduce.T)
data_re = np.matmul(z, v_reduce)
# 可视化
plt.plot(data[:, 0], data[:, 1], '.b')  # 原始样本
plt.plot([v_reduce[0][0], 0], [v_reduce[0][1], 0], '--r')  # 特征向量
plt.plot(data_re[:, 0], data_re[:, 1], 'or')  # 压缩后再重建输入
plt.show()

# **************** 练习2：图片压缩 ****************
# Dataset共有1000张图片，只用前100个进行训练
data = scio.loadmat('ex7faces.mat')['X'][:100, :]  # 导入数据
data, data_mean = normalize(data)  # 归一化处理

u, v, k = cap.pca(data)  # 执行PCA算法

# # 用左奇异矩阵U来降维（行降维）
# u_reduce = u[:, :36]  # 用U的“前k列”作为主成分
# z = np.matmul(u_reduce.T, data)  # 得到图片压缩后的表示z（此形式不能可视化为图片）
# data_re = np.matmul(u_reduce, z)  # 通过压缩后的z，来重建输入data_re

# 用右奇异矩阵V来降维（列降维）
v_reduce = v[:k, :]  # 用V的“前k行”作为主成分（可调节此值，观察重建效果的变化）
z = np.matmul(data, v_reduce.T)
data_re = np.matmul(z, v_reduce)

# 可视化
# display(u[:, :36].T, [32, 32], [6, 6])  # 可视化u的前几个主成分
display(v[:36, :], [32, 32], [6, 6])      # 可视化v的前几个主成分
display(data, [32, 32], [10, 10])     # 原图
display(data_re, [32, 32], [10, 10])  # 重建

# 测试未经训练的图片能不能压缩并重建
test_images = scio.loadmat('ex7faces.mat')['X'][100:105, :]  # 新的图片集
test_images_norm = test_images - data_mean  # 用训练集的mean归一化处理
test_z = np.matmul(test_images_norm, v_reduce.T)  # 用训练集学习到的特征对测试集降维
test_re = np.matmul(test_z, v_reduce)  # 重建

# 依次展示：测试原图、测试原图用训练集mean归一化后、重建后图像
show = np.row_stack((test_images, test_images_norm))
show = np.row_stack((show, test_re))
display(show, [32, 32], [3, 5])
# 测试结果：重建效果还行，但是不算好，肉眼还是能分辨出前后差异，但是对于训练
#         过的图片，重建效果就很好，肉眼分辨不出差异，说明PCA泛化能力不高。











