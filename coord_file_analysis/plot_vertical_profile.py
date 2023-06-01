
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
dir_label = '00020'
res = 200
filename = "my_experiment00000.csv"
first_part_of_path = '/nobackup/scgf/myExperiments/gaussJan2022/gj170/'

dir = first_part_of_path+'size'+str(dir_label)        # res200


def read_calculate_plot(filename):
    """
    read the csv file, plot the difference between rows (diff in y coordinates)
    """
    myExp = pd.read_csv(filename, header=0)
    df_v = myExp[50:len(myExp):200]    # dataframe only containing a vertical line. start from the 50th element and skip 2 rows of 200 elements
    

    ycoord_vals = df_v["y coord"].values
    porosity_vals = df_v["Porosity"].values

    x = range(0,len(df_v),1)
    plt.plot(porosity_vals,x)
    plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)  # to avoid offset in plot 
    plt.show()


print("\n",dir)
os.chdir(dir)

myfile = Path(os.getcwd()+'/'+filename)  # build file name including path
if myfile.is_file():
    read_calculate_plot(filename)
    