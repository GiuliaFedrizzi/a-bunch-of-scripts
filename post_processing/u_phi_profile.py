"""
plot u*phi as a measure of flux leaving the domain
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path



res = 200    # resolution in the x direction
res_y = 230  # resolution in the y direction
filename = "my_experiment16000.csv"
first_part_of_path = '/nobackup/scgf/myExperiments/threeAreas/prod/prt/prt45/rt0.5/visc_3_1e3/vis1e3_mR_08'


dir = first_part_of_path

# define the indexes for the two profiles, the first one is the start for the horizontal, the second for the vertical profile
# point_indexes = [res*res_y-res,5]   # X original: res*res_y/2+res, central: res*res_y/2 (?)
point_indexes = [(res*res_y*0.75+res),5]   # 3/4 of the domain
print(f'index for the horizontal profile: {point_indexes[0]}')
def read_calculate_plot(filename, var_to_plot,point_indexes):

    myExp = pd.read_csv(filename, header=0)
    # df_v = myExp[50:len(myExp):res]    # dataframe only containing a vertical line. start from the 50th element and skip 2 rows of 200 elements
    df_v = myExp[int(point_indexes[1]):len(myExp):int(res)]    # dataframe only containing a vertical line. start from the n-th element and skip 2 rows of 200 elements
    hor_index_start = int(point_indexes[0])  # index for the horizontal profile (constant y)
    df_h = myExp[hor_index_start:hor_index_start+res:1]    # dataframe only containing a horizontal line. start from the left boundary, continue for 200 elements
    # print(len(df_v))
    variable_vals_v = df_v[var_to_plot].values
    variable_vals_h = df_h[var_to_plot].values
    x_v = np.arange(0,1,1/len(df_v))
    x_h = np.arange(0,1,1/len(df_h))
    return (x_v, variable_vals_v), (x_h, variable_vals_h)


os.chdir(dir)
myfile = Path(os.getcwd()+'/'+filename)  # build file name including path


fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

vars_to_plot = ["x velocity","y velocity", "Porosity"]


# prepare to store vertical and horizontal data
all_data_v = {}
all_data_h = {}

for v in vars_to_plot:
    (x_v, y_v), (x_h, y_h) = read_calculate_plot(filename, v,point_indexes)
    all_data_v[v] = (x_v, y_v)
    all_data_h[v] = (x_h, y_h)
# print(f'all_data_v \n{all_data_v}')
if False:
    """
    plot velocity and porosity (2 lines) on the same plots, one horizontal, one vertical
    """
    # Plotting on ax1: y velocity - horizontal profile of vertical velocity
    x_vel_y, y_vel_y = all_data_h[vars_to_plot[1]]  # y velocity
    x_poro, y_poro = all_data_h[vars_to_plot[2]]
    ax1a = ax1.twinx()
    print(np.corrcoef(abs(y_vel_y),y_poro)) 

    line1_h, = ax1.plot(x_vel_y, y_vel_y)
    line2_h, = ax1a.plot(x_poro, y_poro,'g')  # plot porosity in green

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
    x_vel_x, y_vel_x = all_data_v[vars_to_plot[0]]
    x_poro, y_poro = all_data_v[vars_to_plot[2]]
    ax2a = ax2.twiny()

    line1_v, = ax2.plot(y_vel_x, x_vel_x, label=vars_to_plot[0])
    line2_v, = ax2a.plot(y_poro, x_poro,'g', label=vars_to_plot[2])

    ax2.set_ylabel('y')
    ax2.set_xlabel(vars_to_plot[0])
    ax2a.set_xlabel(vars_to_plot[2], color='g')  # Setting color to match line color
    ax2a.tick_params(axis='x', colors='g') 

    # Legend
    ax2.legend([line1_v, line2_v], 
            [vars_to_plot[0], vars_to_plot[2]],loc=(0.8, 0.9))

    plt.show()

if True:
    """
    plot the product: u*phi
    """
    x_vel_y, y_vel_y = all_data_h[vars_to_plot[1]] 
    x, y_poro = all_data_h[vars_to_plot[2]]

    line1_h, = ax1.plot(x, y_vel_y*y_poro)
    ax1.set_xlabel('x') 
    ax1.set_ylabel('$\phi * u_y$') 

    x_vel_x, y_vel_x = all_data_v[vars_to_plot[0]] 
    x, y_poro = all_data_v[vars_to_plot[2]]

    line1_v, = ax2.plot(y_vel_x*y_poro,x)
    ax2.set_xlabel('$\phi * u_x$') 
    ax2.set_ylabel('y') 
    plt.show()
