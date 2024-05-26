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
from useful_functions import extract_two_profiles,getResolution



res = getResolution()    # resolution in the x direction
res_y = int(res*1.15)  # resolution in the y direction
# filename = "my_experiment16000.csv"
filenames = ["my_experiment11000.csv", "my_experiment12000.csv", "my_experiment13000.csv","my_experiment14000.csv", "my_experiment15000.csv","my_experiment16000.csv"]
dir_path = '/nobackup/scgf/myExperiments/threeAreas/prod/prt/prt45/rt0.5/visc_3_1e3/vis1e3_mR_08'


# define the indexes for the two profiles, the first one is the start for the horizontal, the second for the vertical profile
# REMEMBER to remove 1 from the horizontal index
point_indexes = [(res*res_y-(5*res))-1,5]   # top (horizontal) and left (vertical). X original: res*res_y/2+res, central: res*res_y/2 (?)
# point_indexes = [(res*res_y*0.75+res),5]   # 3/4 of the domain
# print(f'index for the horizontal profile: {point_indexes[0]}')


os.chdir(dir_path)


fig, ax1 = plt.subplots(nrows=1, ncols=1)

vars_to_plot = ["x velocity","y velocity", "Porosity"]

int_ratios = []

for filename in filenames:
    print(filename)
    # prepare to store vertical and horizontal data
    all_data_v = {}
    all_data_h = {}

    for v in vars_to_plot:
        (x_v, y_v), (x_h, y_h) = extract_two_profiles(filename, v,point_indexes)
        all_data_v[v] = (x_v, y_v)
        all_data_h[v] = (x_h, y_h)
        

    x_coord_vel_hor, y_vel_hor = all_data_h[vars_to_plot[1]] 
    x, poro_values_hor = all_data_h[vars_to_plot[2]]

    integral_hor = np.trapz(abs(y_vel_hor), x=x_coord_vel_hor)
    print(f'integral_hor {integral_hor}')

    y_coord_vel_ver, x_vel_ver = all_data_v[vars_to_plot[0]] 
    x, poro_values_ver = all_data_v[vars_to_plot[2]]
    integral_ver =  np.trapz(abs(x_vel_ver*poro_values_ver), x = y_coord_vel_ver)

    int_ratio = integral_hor/integral_ver

    print(f'integral_ver {integral_ver}')
    print(f'integral ratio {int_ratio}')
    int_ratios.append(int_ratio)

ax1.plot(range(len(int_ratios)),int_ratios,'o-')

plt.show()

