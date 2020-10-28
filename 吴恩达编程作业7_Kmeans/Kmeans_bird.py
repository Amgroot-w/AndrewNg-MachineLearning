# k-means 实战 -- Image Compression 图像压缩（颜色）
# Feb.12th

import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import matplotlib.image as maping
from mpl_toolkits.mplot3d import Axes3D
from K_means import kmeans
import cap

# # 练习1：3类数据进行聚类
# data = scio.loadmat('ex7data2.mat')['X']  # 导入数据
# [c, centroid] = kmeans(data, K=3, iteration=10)  # 3类，迭代不超过10次即满足结束条件

# 练习2：bird-图像压缩
bird = maping.imread('bird_small.png')  # 导入数据
data = bird.reshape(bird.shape[0]*bird.shape[1], bird.shape[2])  # 变换为二维数组

[c, centroid] = cap.kmeans(data, K=16, iteration=10)  # K-means聚类

data_re = np.zeros(data.shape)
for i in range(data.shape[0]):
    data_re[i] = centroid[c[i]]  # 以聚类中心代替原样本
bird_re = data_re.reshape([bird.shape[0], bird.shape[1], bird.shape[2]])  # 二维数组还原为RGB图像

# 处理前后对比图
plt.imshow(bird)     # 处理前原图
plt.title('Before')
plt.show()
plt.imshow(bird_re)  # 压缩后图像
plt.title('After')
plt.show()






