# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 16:21:39 2022

@author: PURUSHOT
"""

import numpy as np 

import matplotlib.pyplot as plt

data_arr = np.load(r"C:\Users\purushot\Desktop\LaueNN_script\GaN_Si_2phase\training_data\GaN_Si_grain_2_859_nb11.npz")

codebars = data_arr["arr_0"]
location = data_arr["arr_1"]
ori_mat = data_arr["arr_2"]
ori_mat1 = data_arr["arr_3"]
flag = data_arr["arr_4"]
s_tth = data_arr["arr_5"]
s_chi = data_arr["arr_6"]
s_miller_ind = data_arr["arr_7"]

plt.scatter(s_tth, s_chi)