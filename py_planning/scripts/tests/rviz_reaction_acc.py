import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import math

import matplotlib
import matplotlib.pyplot as plt
import paper_plot.utils as plot_utils
import paper_plot.functions as plot_func

##############################################################
# visualization

fig = plt.figure()
plot_utils.fig_reset()

# ax1
fig_ax1 = fig.add_subplot(111)
plot_utils.subfig_reset()
plot_utils.axis_set_xlabel(fig_ax1, 't (s)')
plot_utils.axis_set_ylabel(fig_ax1, 'acc (m/$s^2$)')

coefs = [0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0]
min_t = 0.0
max_t = 6.0
min_acc = -4.0
max_acc = -2.0

t = np.arange(min_t, (max_t+0.01), 0.1)
t[t < min_t] = min_t
t[t > max_t] = max_t

for coef in coefs:
  dist_norm = np.power(max_t - min_t, coef)
  dist2_min_t = np.power((t - min_t), coef)
  dist2_max_t = np.power((max_t - t), coef)

  # acc = 1.0 - (2.0 / (1 + np.exp(coef * t)) - 1.0) / (2.0 / (1 + np.exp(coef * max_t)) - 1.0)
  acc = 0.0 * dist2_min_t / dist_norm + 1.0 * dist2_max_t / dist_norm
  acc = acc * (min_acc - 1.0 * max_acc) + max_acc

  print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
  print("coef = {}, with samples = {}".format(coef, t))
  print("and values = {}".format(acc))
  print(" ")
  plt.plot(t, acc, label="coef={:.1f}".format(coef))
  plt.legend()

# acc = 1.0 / (1 + np.exp(-t))
# plt.plot(t, acc)

plt.show()