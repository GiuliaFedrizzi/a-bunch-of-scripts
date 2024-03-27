""" 
Save a figure based on broken bonds

made for wavedec2022, to be analysed in imagej

"""

# I use this ^ to run python in VS code in interactive mode
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import glob
import seaborn as sns
from pathlib import Path
import re   # regex
import sys
from matplotlib.lines import Line2D

sys.path.append('/home/home01/scgf/myscripts/post_processing')   # where to look for useful_functions.py

from useful_functions import * 
from useful_functions_moments import *
 
# palette_option = ['white','black']  # this one works on my laptop
palette_option = {-1:'w',0:'b'}       # this one works on ARC. Now saves in BLUE!!
palette_option_bb = {0:'b', 1:'w', 2:'w', 3:'w', 4:'w', 5:'w', 6:'w',7:'w'}  # no broken bonds = blue,  bb >=1  = white


def read_calculate_plot(filename,input_tstep):
    """
    read the csv file, plot fractures
    """
    timestep_number = int(re.findall(r'\d+',filename)[0])   # regex to find numbers in string. Then convert to float. Will be used to get "time" once multiplied by timestep.

    bb_df = pd.read_csv(filename, header=0)  # build the dataframe from the csv file

    time = str(input_tstep*timestep_number)
    print(f'time {time}, tstep: {input_tstep}')

    if 'Fractures' in bb_df.columns:
        sns.scatterplot(data=bb_df,x="x coord",y="y coord",hue="Fractures",marker='.',s=9,palette=palette_option,linewidth=0,legend=False).set_aspect('equal') #,alpha=0.8  hue="Fracrures",
    elif 'Fracrures' in bb_df.columns:  # account for a spelling mistake: some old simulations have "Fracrures"
        sns.scatterplot(data=bb_df,x="x coord",y="y coord",hue="Fracrures",marker='.',s=9,palette=palette_option,linewidth=0,legend=False).set_aspect('equal') #,alpha=0.8  hue="Fracrures",
    else:
        sns.scatterplot(data=bb_df,x="x coord",y="y coord",hue="Broken Bonds",marker='.',s=9,palette=palette_option_bb,linewidth=0,legend=False).set_aspect('equal') # I've stopped saving "Fractures"
    plt.title(time)
    plt.tight_layout()
    plt.savefig("py_bb_"+str(timestep_number).zfill(6)+".png",dpi=300)
    plt.clf()
    # plt.show()

print(os.getcwd())

input_tstep = float(getTimeStep("input.txt"))

for filename in sorted(glob.glob("my_experiment*.csv")):
    """ loop through files"""
    print(f'filename: {filename}')

    myfile = Path(os.getcwd()+'/'+filename)  # build file name including path
    if myfile.is_file():
        read_calculate_plot(filename,input_tstep)
    
