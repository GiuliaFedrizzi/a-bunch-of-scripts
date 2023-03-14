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
from matplotlib.lines import Line2D

from useful_functions import * 
from useful_functions_moments import *
 
# palette_option = ['white','black']  # this one works on my laptop
palette_option = {-1:'w',0:'k'}       # this one works on ARC

#dir_labels = ['vis1e2_mR_1']#,'vis1e2_mR_2']
#filenames = ["my_experiment05000.csv","my_experiment05300.csv"]

# first_part_of_path = '/nobackup/scgf/myExperiments/wavedec2022/wd05_visc/visc_4_3e4/'
#first_part_of_path = '/Users/giuliafedrizzi/Library/CloudStorage/OneDrive-UniversityofLeeds/PhD/arc/myExperiments/wavedec2022/wd05_visc/visc_2_1e2/'


def read_calculate_plot(filename,input_tstep):
    """
    read the csv file, plot fractures
    """
    timestep_number = int(re.findall(r'\d+',filename)[0])   # regex to find numbers in string. Then convert to float. Will be used to get "time" once multiplied by timestep.

    bb_df = pd.read_csv(filename, header=0)

    # if len(bb_df)==0:
    #     print("No broken bonds.")
    #     return
    time = str(input_tstep*timestep_number)
    print(f'time {time}, tstep: {input_tstep}')
    plt.figure()
    #sns.scatterplot(data=bb_df,x="xcoord100",y="ycoord100",hue="Broken Bonds",linewidth=0,alpha=0.8,marker="h",size=0.6).set_aspect('equal')
    sns.scatterplot(data=bb_df,x="x coord",y="y coord",hue="Fracrures",marker='.',s=9,palette=palette_option,linewidth=0,legend=False).set_aspect('equal') #,alpha=0.8  hue="Fracrures",
    plt.title(time)
    plt.savefig("py_bb_"+str(timestep_number).zfill(5)+".png",dpi=300)
    # plt.show()
        ##    .set_aspect('equal') to keep the same scale for x and y

print(os.getcwd())
#for i,filename in enumerate(sorted(glob.glob("my_experiment*"))[2:36:5]): #[beg:end:step]  # loop through files
# get the time step used in the simulation (do it once, from the first simulation)
input_tstep = float(getTimeStep("input.txt"))

for filename in sorted(glob.glob("my_experiment*.csv")):
    """ loop through files"""
    print(f'filename: {filename}')

    myfile = Path(os.getcwd()+'/'+filename)  # build file name including path
    if myfile.is_file():
        read_calculate_plot(filename,input_tstep)
    
    #fig.suptitle("t = "+str('{:.1e}'.format(input_tstep*timestep_number))) 
    plt.tight_layout()
    #fig_name = first_part_of_path+''.join(dir_labels)+"t"+str(timestep_number)+".png" # join together all elements of the list of sizes (directories)
    #plt.savefig(fig_name, dpi=600)#,transparent=True)
    # plt.show()
