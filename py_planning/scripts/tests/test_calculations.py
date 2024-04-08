import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import heapq
import copy

##################################################
# a = np.array([3.0, 0.6, 1.0])
# b = np.array([1.0, 0.5, 0.5])
# print(np.floor(a / b))
# print(np.sum(np.logical_or(a < b, a == b)))

##################################################
# test_dict = {}
# test_dict[(1, 2)] = 'hellow world'
# test_dict[(1, 2.1)] = 'bye world'
# print(test_dict)

##################################################
# inputs = np.array([[1, 2, 3], [4, 5, 6], [2, 2, 2]], dtype=int)
# checks = np.sum(np.logical_or(inputs < 0, inputs > np.array([5, 5, 5])), axis=1)

# for input in inputs[checks < 1e-3]:
#     print(input)

##################################################
# test_list = []
# heapq.heappush(test_list, (0.5, ("D", 0)))
# heapq.heappush(test_list, (0.1, ("A", 0)))
# heapq.heappush(test_list, (0.7, ('E', 0)))
# heapq.heappush(test_list, (0.2, ['B', 0]))
# heapq.heappush(test_list, (0.3, ('C', 0)))

# rlist = copy.copy(test_list)
# print("before reverse", rlist)
# rlist.reverse()
# print("reverse list", rlist)

# dvalue = (0.5, ("D", 0))
# if dvalue in test_list:
#     test_list.remove(dvalue)

# while test_list:
#     priority, value = heapq.heappop(test_list)
#     print(priority, value)

##################################################

# check = [-1.0, 2.0, 1.0, 3.0]
# check = sorted(check)
# print(check)
print(np.arange(1, 1))
