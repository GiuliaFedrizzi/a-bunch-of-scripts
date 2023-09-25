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
filename = "my_experiment00100.csv"
filename_fluid = "fluidgrid00100.csv"
first_part_of_path = '/nobackup/scgf/myExperiments/smooth/sm75/'

dir = first_part_of_path+'size'+str(dir_label)        # res200


def read_calculate_plot(filename,filename_fluid):
    """
    read the csv file, plot the difference between rows (diff in y coordinates)
    """
    myExp = pd.read_csv(filename, header=0)
    myExp_f = pd.read_csv(filename_fluid, header=0)
    # df_v = myExp[50:len(myExp):res]    # dataframe only containing a vertical line. start from the 50th element and skip 2 rows of 200 elements
    df_v = myExp[50:len(myExp):int(res/2)]    # dataframe only containing a vertical line. start from the 50th element and skip 2 rows of 200 elements
    df_v_f = myExp_f[25:len(myExp_f):int(res/2)]    # dataframe only containing a vertical line. start from the 50th element and skip 2 rows of 200 elements
    

    # ycoord_vals = df_v["y coord"].values
    # ycoord_vals_f = df_v_f["y coord"].values
    porosity_vals = df_v["Porosity"].values
    porosity_vals_f = df_v_f["Porosity"].values

    x = np.arange(0,1,1/len(df_v))
    x_f =np.arange(0,1,1/len(df_v_f))
    plt.plot(porosity_vals,x)
    # plt.plot(porosity_vals_f,x_f)
    plt.xlabel("Porosity")
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
    