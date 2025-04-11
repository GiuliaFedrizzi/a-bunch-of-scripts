"""
Plot porosity and pressure profile (horizontal)
To show porosity and fluid pressure boundary conditions
"""



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import glob
# import seaborn as sns
from pathlib import Path


# options to set:
res = 200 # horizontal resolution = n of solid particles in one row (can be 200 or 400)
res_y = 230  # vertical resolution (can be 230 or 460)
filename = "my_experiment00006.csv"
first_part_of_path = '/mnt/scratch/pdlw636/myExperiments/threeAreas/prod/pb54/visc_1_1e1/'  # to show boundary conditions
# first_part_of_path = '/nobackup/scgf/myExperiments/threeAreas/prod/prt/singleInjection/si25/'
#  Pressure   Porosity
my_dir = first_part_of_path+'vis1e1_mR_02'       
# poro_colour = (161/255, 62/255, 22/255)
poro_colour = ('g')
# press_colour = (0/255, 28/255, 148/255)  #1f77b4
press_colour = ('#1f77b4')  # default blue  

def read_calculate_plot(filename, var_to_plot):

    myExp = pd.read_csv(filename, header=0)
    # print(myExp.idxmax())#(axis=""))
    df_v = myExp[50:len(myExp):res]    # dataframe only containing a vertical line. start from the 50th element and skip 2 rows of 200 elements
    # df_v = myExp[0:len(myExp):int(res)]    # dataframe only containing a vertical line. start from the n-th element and skip 2 rows of 200 elements
    df_h = myExp[int(res*res_y/2+2*res-1):int(res*res_y/2+3*res-1):1]    # dataframe only containing a horizontal line. start from the 50th element and skip 2 rows of 200 elements
    # -1 because the first particle (bottom left corner) is missing, so everything needs to be shifted

    variable_vals_v = df_v[var_to_plot].values
    variable_vals_h = df_h[var_to_plot].values
    x_v = np.arange(0,1,1/len(df_v))
    x_h = np.arange(0,1,1/len(df_h))
    return (x_v, variable_vals_v), (x_h, variable_vals_h)

os.chdir(my_dir)
myfile = Path(os.getcwd()+'/'+filename)  # build file name including path

plt.rcParams["font.weight"] = "bold"  # set default weight to bold
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

fig, ax1 = plt.subplots(nrows=1, ncols=1,figsize=(9,6))

vars_to_plot = ["Pressure", "Porosity"]

# prepare to store horizontal data
all_data_h = {}

for v in vars_to_plot:
    (x_v, y_v), (x_h, y_h) = read_calculate_plot(filename, v)
    all_data_h[v] = (x_h, y_h)

# Plotting on ax1 
x, y = all_data_h[vars_to_plot[0]]
line1_h, = ax1.plot(x, y, label=f"{vars_to_plot[0]} (horizontal)",color=press_colour,linewidth=3)
ax1.yaxis.set_major_locator(plt.MaxNLocator(4))  # reduce the number of annotations on the y axis
# ax1.set_xlim([0.35,0.65])

# ax1.set_ylim(bottom=5.885e7)


# Set labels for ax1 
ax1.set_ylabel(vars_to_plot[0],color=press_colour,fontsize=12,weight='bold')
# ax1.set_xlabel("x")
ax1.tick_params(axis='y', colors=press_colour,labelsize=12)
ax1.tick_params(axis='x', labelsize=12)
# ax1.set_title("res*res_y/2+2*res-1")

# after plotting the data, format the labels
current_values = plt.gca().get_yticks()
plt.gca().set_yticklabels(['{:.2e}'.format(x) for x in current_values])

# porosity - horizontal
ax1a = ax1.twinx()
x, y = all_data_h[vars_to_plot[1]]
print(f'len x coord {len(y)}')
line2_h, = ax1a.plot(x, y, label=f"{vars_to_plot[1]} (horizontal)", color=poro_colour,linewidth=3)
ax1a.set_ylabel(vars_to_plot[1], color=poro_colour,fontsize=12,weight='bold')  # Setting color to match line color
ax1a.yaxis.set_major_locator(plt.MaxNLocator(4))  # reduce the number of annotations on the y axis
# ax1a.set_ylim(top=0.1997)
ax1a.tick_params(axis='y', colors=poro_colour)
ax1a.yaxis.tick_right()  # Ensure the y-axis label is on the right
ax1a.get_xaxis().get_major_formatter().set_useOffset(False)


# Adjusting positions to avoid overlap
ax1.yaxis.tick_left()


# Legend
# fig.legend([line1_v, line2_v], 
#            [vars_to_plot[0], vars_to_plot[1]],loc=(0.8, 0.8))


ax1.spines[['right', 'top']].set_visible(False)  # remove plot margins pressure
ax1a.spines[['left', 'top']].set_visible(False)  # remove plot margins  poro

# after plotting the data, format the labels
current_values = plt.gca().get_yticks()
plt.gca().set_yticklabels(['{:.3f}'.format(x) for x in current_values])
plt.show()
# plt.savefig("poro_press_central_point_poro.png",dpi=400,transparent=True)
print("Done")
# """
