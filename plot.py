# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 17:35:30 2020

@author: Yuxuan Long
"""

import numpy as np
import scipy.linalg as lin
import matplotlib.pyplot as plt
          
          
n_list = [40, 50, 60, 70, 80, 84, 90, 100, 150, 200, 250, 300, 350, 370]

direct_t = [16, 85, 361, 1219, 2900, 3603]

lr_B1_t = [4, 10, 29, 46, 62, 61, 97, 99, 300, 810, 1631, 2222, 4007]

lr_B2_t = [2, 7, 15, 24, 45, 39, 63, 82, 283, 538, 1080, 1609, 2620, 4082]

plt.plot(n_list[0:6], direct_t, label="Direct")
plt.plot(n_list[:-1], lr_B1_t, label="ALS B1")
plt.plot(n_list, lr_B2_t, label="ALS B2")

plt.plot(np.ones((4050, )) * 84, np.arange(-50, 4000), '--')


n_dense = np.arange(40, 350, 1)
pred_t = (n_dense ** 3) / 10000

plt.plot(n_dense, pred_t, '--')

plt.legend()
# plt.title("")
plt.xlabel("n")
plt.ylabel("Time taken (seconds)")
plt.show()

# plt.figure()
# rank_B1 = [32, 35, 38, 42, 42, 42, 45, 44, 50, 59, 64, 70, 76]

# rank_B2 = [30, 32, 35, 37, 40, 39, 41, 40, 50, 54, 63, 69, 75, 79]

# start_rank = np.round(np.array(n_list) / (np.log2(np.array(n_list) / 25) + 1))

# plt.plot(n_list[:-1], rank_B1, label = "B1")
# plt.plot(n_list, rank_B2, label = "B1")
# plt.plot(n_list, start_rank, label = "Start")

# plt.legend()
# # plt.title("")
# plt.xlabel("n")
# plt.ylabel("Rank")
# # plt.gca().set_aspect('equal', adjustable='box')
# plt.show()