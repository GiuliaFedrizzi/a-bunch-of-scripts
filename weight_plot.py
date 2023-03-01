#!/home/home01/scgf/.conda/envs/condaPandas/bin/python3

""" plots the variables "Weight" and "Move"
version for gaussJan2022/gj*
"""
# goes through multiple directories
import matplotlib.pyplot as plt
import os as os
import numpy as np

# import functions from external file
from useful_functions import * 


dir_labels = ['00200','00400','00600','00800','01000','02000','04000','08000','10000']
weight = np.zeros(len(dir_labels))
move = np.zeros(len(dir_labels))

parent_directories = []
first_part_of_path = '/nobackup/scgf/myExperiments/gaussJan2022/gj25/size'

for i in dir_labels:
    parent_directories.append(first_part_of_path+str(i))

fig, axs = plt.subplots(nrows=1, ncols=2)  # 2 subplots, 1 for weight, 1 for move


for a,parent_directory in enumerate(parent_directories):
    inputFile = parent_directory + "/latte.log"
    print(inputFile)
    weight[a] = float(getParameterFromLatte(inputFile,"Weight")) # find "Weight" in latte
    move[a] = float(getParameterFromLatte(inputFile,"Move"))

dir_list_num = [float(l) for l in dir_labels]  # convert from strings to numerical values with list comprehension

axs[0].plot(dir_list_num,weight, 'o',linewidth=1,markersize=2)
axs[0].set_title("Weight")
axs[0].grid(linestyle='--',alpha=0.6)
    
axs[1].plot(dir_list_num,move, 'o',linewidth=1,markersize=2)
axs[1].set_title("Move")
axs[1].grid(linestyle='--',alpha=0.6)

fig_title = "Weight, Move\n " + first_part_of_path + "*"

# #axs.set_xscale('log')
# #plt.savefig(figure_name, dpi=600)
# plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
# plt.legend()
plt.show()

