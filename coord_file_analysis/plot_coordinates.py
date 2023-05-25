
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
dir_label = '00400'
res = 200
filename = "my_experiment-0003.csv"
first_part_of_path = '/nobackup/scgf/myExperiments/gaussJan2022/gj164/'

dir = first_part_of_path+'size'+str(dir_label)        # res200


def read_calculate_plot(filename):
    """
    read the csv file, plot
    """
    myExp = pd.read_csv(filename, header=0)
    df_v = myExp[50:len(myExp):400]    # dataframe only containing a vertical line. start from the 50th element and skip 2 rows of 200 elements

    print(df_v)
    
    df_v["dy"] = 0

    ycoord_vals = df_v["y coord"].values
    print(ycoord_vals)
    diff = ycoord_vals[1:] - ycoord_vals[:-1]
    print(diff)

    x = range(0,len(diff),1)
    plt.plot(diff,x)
    plt.show()


print("\n",dir)
os.chdir(dir)

myfile = Path(os.getcwd()+'/'+filename)  # build file name including path
if myfile.is_file():
    read_calculate_plot(filename)
    