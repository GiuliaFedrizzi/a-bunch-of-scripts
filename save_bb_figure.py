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
 
dir_labels = ['vis1e2_mR_1']#,'vis1e2_mR_2']
filenames = ["my_experiment05000.csv"]# ,"my_experiment05300.csv"]

dir_list = []
# first_part_of_path = '/nobackup/scgf/myExperiments/wavedec2022/wd05_visc/visc_2_1e2/'
first_part_of_path = '/Users/giuliafedrizzi/Library/CloudStorage/OneDrive-UniversityofLeeds/PhD/arc/myExperiments/wavedec2022/wd05_visc/visc_2_1e2/'

Line2D.markers.items() 
#mark_styles = Line2D.filled_markers
#mark_styles = Line2D.markers.keys()
mark_styles = ['o','.']
mark_size = np.arange(0.0001,0.003,0.0005)

# print(Line2D.markers.items())
# print(Line2D.markers.keys())

for i in dir_labels:
    dir_list.append(first_part_of_path+str(i))  


def read_calculate_plot(filename):
    """
    read the csv file, plot fractures
    """
    bb_df = pd.read_csv(filename, header=0)
    #mask = myExp['Broken Bonds'] > 0    # create a mask: find the true value of each row's 'xcoord' column being greater than 0, then using those truth values to identify which rows to keep
    #bb_df = myExp[mask].copy()  # keep only the rows found by mask

    # bb_df.reset_index(drop=True, inplace=True)

    # if len(bb_df)==0:
    #     print("No broken bonds.")
    #     return
    for ms in mark_size:
        plt.figure()
        #sns.scatterplot(data=bb_df,x="xcoord100",y="ycoord100",hue="Broken Bonds",linewidth=0,alpha=0.8,marker="h",size=0.6).set_aspect('equal')
        sns.scatterplot(data=bb_df,x="x coord",y="y coord",hue="Fracrures",palette=['white','black'],linewidth=0,marker='.',size=ms,legend=False).set_aspect('equal') #,alpha=0.8  hue="Fracrures",
        plt.title(ms)
        plt.savefig("python_bb_"+str(ms)+".png",dpi=200)
    # plt.show()
        ##    .set_aspect('equal') to keep the same scale for x and y
    

# get the time step used in the simulation (do it once, from the first simulation)
input_tstep = float(getTimeStep(dir_list[0]+"/input.txt")) 

for dir in dir_list:
    """ loop through directories"""
    os.chdir(dir)
    #for i,filename in enumerate(sorted(glob.glob("my_experiment*"))[2:36:5]): #[beg:end:step]  # loop through files
    for filename in filenames:
        """ loop through files"""
        timestep_number = int(re.findall(r'\d+',filename)[0])   # regex to find numbers in string. Then convert to float. Will be used to get "time" once multiplied by timestep.
    
        myfile = Path(os.getcwd()+'/'+filename)  # build file name including path
        if myfile.is_file():
            read_calculate_plot(filename)
        
        #fig.suptitle("t = "+str('{:.1e}'.format(input_tstep*timestep_number))) 
        print("---filename: ",filename,", timestep_number:",timestep_number,", input timestep: ",input_tstep,", input_tstep*timestep_number: ",input_tstep*timestep_number)
        plt.tight_layout()
        #fig_name = first_part_of_path+''.join(dir_labels)+"t"+str(timestep_number)+".png" # join together all elements of the list of sizes (directories)
        #plt.savefig(fig_name, dpi=600)#,transparent=True)
        # plt.show()
