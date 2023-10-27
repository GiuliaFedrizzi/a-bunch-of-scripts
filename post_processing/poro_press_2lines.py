"""
Plot porosity profile (vertical).
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
filename = "my_experiment00006.csv"
first_part_of_path = '/nobackup/scgf/myExperiments/threeAreas/prod/pb47/visc_1_1e1/'
#  Pressure   Porosity
dir = first_part_of_path+'vis1e1_mR_01'        # res200



def read_calculate_plot(filename, var_to_plot):
    # Assuming this function reads and calculates data based on the filename and variable,
    # returning the calculated x and y values for the given variable.
    
    # ... [some code to read and calculate]

    # Instead of plotting here, return the x and y values.

    myExp = pd.read_csv(filename, header=0)
    # df_v = myExp[50:len(myExp):res]    # dataframe only containing a vertical line. start from the 50th element and skip 2 rows of 200 elements
    df_v = myExp[0:len(myExp):int(res)]    # dataframe only containing a vertical line. start from the 50th element and skip 2 rows of 200 elements
    print(len(df_v))
    variable_vals = df_v[var_to_plot].values
    x = np.arange(0,1,1/len(df_v))
    return x, variable_vals

os.chdir(dir)
myfile = Path(os.getcwd()+'/'+filename)  # build file name including path

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

vars_to_plot = ["Pressure", "Porosity"]
all_data = {}  # store x and y values for each variable

for v in vars_to_plot:
    x, y = read_calculate_plot(filename, v)
    all_data[v] = (x, y)

# Plotting the first variable on ax2
x, y = all_data[vars_to_plot[0]]
line1, = ax2.plot(y, x, label=vars_to_plot[0])
ax2.set_ylabel("y")
ax2.set_xlabel(vars_to_plot[0])
ax2.get_xaxis().get_major_formatter().set_useOffset(False)

# Plotting the second variable using twinx
ax2a = ax2.twiny()
x, y = all_data[vars_to_plot[1]]
line2, = ax2a.plot(y, x, label=vars_to_plot[1], color='r')  # choosing a different color for distinction
ax2a.set_xlabel(vars_to_plot[1])
# If you want a different y label for the second variable:
# ax2a.set_ylabel("Some other label")

# To create a single legend for both lines
fig.legend([line1, line2], [vars_to_plot[0], vars_to_plot[1]], loc='upper right')

plt.show()
