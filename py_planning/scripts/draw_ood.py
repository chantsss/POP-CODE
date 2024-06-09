import numpy as np

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt


# generate two Gaussian distributions
mu1, sigma1 = 0, 0.5 # mean and standard deviation
data1 = np.random.normal(mu1, sigma1, 1000)

mu2, sigma2 = 2, 1 
data2 = np.random.normal(mu2, sigma2, 1000)

plt.hist(data1, bins=50, alpha=0.5, label='Training data')
plt.hist(data2, bins=50, alpha=0.5, label='Actual data')
plt.legend(loc='upper right')
plt.savefig('OOD.png', bbox_inches='tight')
plt.show()