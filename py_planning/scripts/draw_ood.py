import numpy as np

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt


# 生成第一个高斯分布数据
mu1, sigma1 = 0, 0.5 # 均值和标准差
data1 = np.random.normal(mu1, sigma1, 1000)

# 生成第二个高斯分布数据
mu2, sigma2 = 2, 1 # 均值和标准差
data2 = np.random.normal(mu2, sigma2, 1000)

# 绘制两个高斯分布的直方图
plt.hist(data1, bins=50, alpha=0.5, label='Training data')
plt.hist(data2, bins=50, alpha=0.5, label='Actual data')
plt.legend(loc='upper right')
plt.savefig('OOD.png', bbox_inches='tight')
plt.show()