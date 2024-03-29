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
dir_label = '08000'
res = 200
filename = "my_experiment00043.csv"
filename_fluid = "fluidgrid00100.csv"
first_part_of_path = '/nobackup/scgf/myExperiments/threeAreas/prod/pb50/visc_1_1e1/'
var_to_plot = "Pressure"
#  Pressure   Porosity
# dir = first_part_of_path+'size'+str(dir_label)        # res200
dir = first_part_of_path+'vis1e1_mR_01'        # res200


def read_calculate_plot(filename,filename_fluid):
    """
    read the csv file, plot the difference between rows (diff in y coordinates)
    """
    myExp = pd.read_csv(filename, header=0)
    # df_v = myExp[50:len(myExp):res]    # dataframe only containing a vertical line. start from the 50th element and skip 2 rows of 200 elements
    df_v = myExp[100:len(myExp):int(res)]    # dataframe only containing a vertical line. start from the 50th element and skip 2 rows of 200 elements
    print(len(df_v))
    variable_vals = df_v[var_to_plot].values
    x = np.arange(0,1,1/len(df_v))
    if os.path.isfile(filename_fluid):
        myExp_f = pd.read_csv(filename_fluid, header=0)
        df_v_f = myExp_f[25:len(myExp_f):int(res/2)]    # dataframe only containing a vertical line. start from the 50th element and skip 2 rows of 200 elements
        porosity_vals_f = df_v_f[var_to_plot].values
        x_f =np.arange(0,1,1/len(df_v_f))

    # ycoord_vals = df_v["y coord"].values
    # ycoord_vals_f = df_v_f["y coord"].values

    plt.plot(variable_vals,x)
    # plt.plot(porosity_vals_f,x_f)
    plt.xlabel(var_to_plot)
    plt.ylabel("y")
    plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)  # to avoid offset in plot 
    plt.show()


print("\n",dir)
os.chdir(dir)

myfile = Path(os.getcwd()+'/'+filename)  # build file name including path
if myfile.is_file():
    read_calculate_plot(filename,filename_fluid)
else:
    print("No file")
    