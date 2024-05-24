"""
plot u*phi as a measure of flux leaving the domain
"""
# import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import sys
sys.path.append('/home/home01/scgf/myscripts/post_processing')
from useful_functions import extract_two_profiles,getResolution



res = getResolution()    # resolution in the x direction
res_y = int(res*1.15)  # resolution in the y direction
filename = "my_experiment16000.csv"
dir_path = '/nobackup/scgf/myExperiments/threeAreas/prod/prt/prt45/rt0.5/visc_3_1e3/vis1e3_mR_08'

# define the indexes for the two profiles, the first one is the start for the horizontal, the second for the vertical profile
# REMEMBER to remove 1 from the horizontal index
point_indexes = [(res*res_y-(5*res))-1,5]   # top (horizontal) and left (vertical). X original: res*res_y/2+res, central: res*res_y/2 (?)
# point_indexes = [(res*res_y*0.75+res),5]   # 3/4 of the domain
print(f'index for the horizontal profile: {point_indexes[0]}')


os.chdir(dir_path)
myfile = Path(os.getcwd()+'/'+filename)  # build file name including path


fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

vars_to_plot = ["x velocity","y velocity", "Porosity","x coord","y coord","id"]


# prepare to store vertical and horizontal data
all_data_v = {}
all_data_h = {}

for v in vars_to_plot:
    (x_v, y_v), (x_h, y_h) = extract_two_profiles(filename, v,point_indexes)
    all_data_v[v] = (x_v, y_v)
    all_data_h[v] = (x_h, y_h)
# print(f'all_data_v \n{all_data_v}')
if True:
    """
    plot velocity and porosity (2 lines) on the same plots, one horizontal, one vertical
    """
    # Plotting on ax1: y velocity - horizontal profile of vertical velocity
    x_coord_vel_hor, y_vel_hor = all_data_h[vars_to_plot[1]]  # y velocity
    x_coord_poro_hor, poro_values_hor = all_data_h[vars_to_plot[2]]
    ax1a = ax1.twinx()
    print(np.corrcoef(abs(y_vel_hor),poro_values_hor)) 

    line1_h, = ax1.plot(x_coord_vel_hor, y_vel_hor)
    line2_h, = ax1a.plot(x_coord_poro_hor, poro_values_hor,'g')  # plot porosity in green

    # Legend
    ax1.legend([line1_h, line2_h], 
            [vars_to_plot[1], vars_to_plot[2]],loc=(0.6, 0.9))   # labels: y velocity and poro

    ax1.set_xlabel('x') 
    ax1.set_ylabel(vars_to_plot[1]) 
    ax1a.set_ylabel(vars_to_plot[2], color='g')  # Setting color to match line color
    ax1a.yaxis.tick_right()  # Ensure the y-axis label is on the right
    # ax1a.set_ylim(top=0.1997)
    ax1a.tick_params(axis='y', colors='g')

    # Plotting on ax2
    y_coord_vel_ver, x_vel_ver = all_data_v[vars_to_plot[0]]  # y coordinate and x velocity for the vertical profile
    y_coord_poro_ver, poro_values_ver = all_data_v[vars_to_plot[2]]
    ax2a = ax2.twiny()

    line1_v, = ax2.plot(x_vel_ver, y_coord_vel_ver, label=vars_to_plot[0])
    line2_v, = ax2a.plot(poro_values_ver, y_coord_poro_ver,'g', label=vars_to_plot[2])

    ax2.set_ylabel('y')
    ax2.set_xlabel(vars_to_plot[0])
    ax2a.set_xlabel(vars_to_plot[2], color='g')  # Setting color to match line color
    ax2a.tick_params(axis='x', colors='g') 

    # Legend
    ax2.legend([line1_v, line2_v], 
            [vars_to_plot[0], vars_to_plot[2]],loc=(0.8, 0.9))

if False:
    """
    plot the product: u*phi
    """
    x_coord_vel_hor, y_vel_hor = all_data_h[vars_to_plot[1]] 
    x, poro_values_hor = all_data_h[vars_to_plot[2]]

    line1_h, = ax1.plot(x, y_vel_hor*poro_values_hor)
    ax1.set_xlabel('x') 
    ax1.set_ylabel('$\phi * u_y$') 

    y_coord_vel_ver, x_vel_ver = all_data_v[vars_to_plot[0]] 
    x, poro_values_hor = all_data_v[vars_to_plot[2]]

    line1_v, = ax2.plot(x_vel_ver*poro_values_hor,x)
    ax2.set_xlabel('$\phi * u_x$') 
    ax2.set_ylabel('y') 

if False:
    """
    plot the coordinates
    """
    x_coord_x_coord, x_coord_value = all_data_h[vars_to_plot[4]] # y coord (not x)
    # x, poro_values_hor = all_data_h[vars_to_plot[2]]

    line1_h, = ax1.plot(x_coord_x_coord, x_coord_value)
    ax1.set_xlabel('x') 
    ax1.set_ylabel('y coord') 

    y_coord_y_coord, y_coord_value = all_data_v[vars_to_plot[3]] 
    # x, poro_values_hor = all_data_v[vars_to_plot[2]]

    line1_v, = ax2.plot(y_coord_value,y_coord_y_coord)
    ax2.set_xlabel('x coord') 
    ax2.set_ylabel('y') 


if False:
    """ plot id """
    x_coord_id_hor, id_hor = all_data_h[vars_to_plot[5]] 
    print(f'len id_hor {len(id_hor)}')
    ax1.scatter(x_coord_id_hor, id_hor)
    ax1.set_xlabel('x') 
    ax1.set_ylabel('id') 


    y_coord_id_ver, id_ver = all_data_v[vars_to_plot[5]] 
    # x, poro_values_hor = all_data_v[vars_to_plot[2]]

    ax2.scatter(id_ver,y_coord_id_ver)
    ax2.set_xlabel('id') 
    ax2.set_ylabel('y') 

plt.show()

