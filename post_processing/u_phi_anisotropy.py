"""
calculate u*phi as a measure of flux leaving the domain.
ratio between y and x direction to get the flux anisotropy

"""

import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import sys
sys.path.append('/home/home01/scgf/myscripts/post_processing')
from useful_functions import extract_two_profiles,getResolution,average_flow_ratio



res = getResolution()    # resolution in the x direction
res_y = int(res*1.15)  # resolution in the y direction

file_tsteps = ['01','03','07','16']
filenames = ["my_experiment"+i+"000.csv" for i in file_tsteps]# ["my_experiment01000.csv", "my_experiment03000.csv", "my_experiment07000.csv","my_experiment16000.csv"]
dir_path = '/nobackup/scgf/myExperiments/threeAreas/prod/prt/prt45/rt0.5/visc_3_1e3/vis1e3_mR_08'


# define the indexes for the two profiles, the first one is the start for the horizontal, the second for the vertical profile
# REMEMBER to remove 1 from the horizontal index
# point_indexes = [(res*res_y-(5*res))-1,5]   # top (horizontal) and left (vertical). X original: res*res_y/2+res, central: res*res_y/2 (?)
# point_indexes = [(res*res_y*0.75+res),5]   # 3/4 of the domain

hor_points = [(res*res_y-(5*res))-1,
    (res*res_y-(10*res))-1,
    (res*res_y-(15*res))-1,
    (res*res_y-(20*res))-1]
ver_points = [5,10,15,20]

os.chdir(dir_path)


fig, ax1 = plt.subplots(nrows=1, ncols=1)
plt.yticks(fontweight='bold')
ax1a = ax1.twinx()


vars_to_plot = ["x velocity","y velocity", "Porosity"]


int_ratios = []
int_hor_list = []
int_vert_list = []

for filename in filenames:
    print(filename)
    average_hor,average_ver,int_ratio = average_flow_ratio(filename,ver_points,hor_points,vars_to_plot)
    int_ratios.append(int_ratio)
    int_hor_list.append(average_hor)
    int_vert_list.append(average_ver)
    
ax1.axhline(y=0.0,color='k',linestyle='--',alpha=0.4)
ax1a.axhline(y=0.0,color='grey',linestyle=':',alpha=0.5)
p1 = ax1.plot(file_tsteps,int_ratios,'o-',linewidth=2,color='black',label='Ratio')
p2 = ax1a.plot(file_tsteps,int_hor_list,'o--',label='Top Vol Flow Rate, $\phi u_y$')
p3 = ax1a.plot(file_tsteps,int_vert_list,'o--',label='Side Vol Flow Rate, $\phi u_x$')
ax1.set_xlabel("File Number")
ax1.set_ylabel("$\mathbf{Q_{top}/Q_{side}}$ Ratio",fontweight='bold')
ax1a.set_ylabel("$\phi u_i $")

plots = p1+p2+p3

labs = [l.get_label() for l in plots]
ax1.legend(plots, labs, loc=0)


plt.show()

