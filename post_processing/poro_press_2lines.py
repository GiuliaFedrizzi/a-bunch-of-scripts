"""
Plot porosity and pressure profile (vertical + horizontal) on the same plot
"""



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import glob
import seaborn as sns
from pathlib import Path


# options to set:
plot_figure = 1
res = 200
res_y = 230
filename = "my_experiment00006.csv"
first_part_of_path = '/nobackup/scgf/myExperiments/threeAreas/prod/pb47/visc_1_1e1/'
#  Pressure   Porosity
dir = first_part_of_path+'vis1e1_mR_01'        # res200



def read_calculate_plot(filename, var_to_plot):

    myExp = pd.read_csv(filename, header=0)
    # print(myExp.idxmax())#(axis=""))
    # df_v = myExp[50:len(myExp):res]    # dataframe only containing a vertical line. start from the 50th element and skip 2 rows of 200 elements
    df_v = myExp[0:len(myExp):int(res)]    # dataframe only containing a vertical line. start from the 50th element and skip 2 rows of 200 elements
    df_h = myExp[int(res*res_y/2+res):int(res*res_y/2+2*res):1]    # dataframe only containing a vertical line. start from the 50th element and skip 2 rows of 200 elements
    # print(len(df_v))
    variable_vals_v = df_v[var_to_plot].values
    variable_vals_h = df_h[var_to_plot].values
    x_v = np.arange(0,1,1/len(df_v))
    x_h = np.arange(0,1,1/len(df_h))
    return (x_v, variable_vals_v), (x_h, variable_vals_h)

os.chdir(dir)
myfile = Path(os.getcwd()+'/'+filename)  # build file name including path


fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

vars_to_plot = ["Pressure", "Porosity"]

# prepare to store vertical and horizontal data
all_data_v = {}
all_data_h = {}

for v in vars_to_plot:
    (x_v, y_v), (x_h, y_h) = read_calculate_plot(filename, v)
    all_data_v[v] = (x_v, y_v)
    all_data_h[v] = (x_h, y_h)

# Plotting on ax1 and ax2
x, y = all_data_h[vars_to_plot[0]]
line1_h, = ax1.plot(x, y, label=f"{vars_to_plot[0]} (horizontal)")

x, y = all_data_v[vars_to_plot[0]]
line1_v, = ax2.plot(y, x, label=vars_to_plot[0])

# Set labels for ax1 and ax2
ax1.set_ylabel(vars_to_plot[0])
ax1.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_xlabel(vars_to_plot[0])

# Disable offsets
ax1.get_xaxis().get_major_formatter().set_useOffset(False)
ax2.get_xaxis().get_major_formatter().set_useOffset(False)

# Create twin axes and plot
ax2a = ax2.twiny()
x, y = all_data_v[vars_to_plot[1]]
line2_v, = ax2a.plot(y, x, label=vars_to_plot[1], color='r')
ax2a.set_xlabel(vars_to_plot[1], color='r')  # Setting color to match line color

ax1a = ax1.twinx()
x, y = all_data_h[vars_to_plot[1]]
line2_h, = ax1a.plot(x, y, label=f"{vars_to_plot[1]} (horizontal)", color='r')
ax1a.set_ylabel(vars_to_plot[1], color='r')  # Setting color to match line color
ax1a.yaxis.tick_right()  # Ensure the y-axis label is on the right

# Adjusting positions to avoid overlap
ax1.yaxis.tick_left()
ax2.yaxis.tick_left()

# Legend
fig.legend([line1_v, line2_v], 
           [vars_to_plot[0], vars_to_plot[1]],loc=(0.8, 0.8))

plt.subplots_adjust(wspace=0.3)  # Adjust space between plots

plt.show()
