""" 
builds matrix:



"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import glob
import seaborn as sns
from pathlib import Path
import re   # regex

from useful_functions import * 
# from useful_functions_moments import *

# options to set:
plot_figure = 1
dir_label = ''
res = 200
filename = "my_experiment32000.csv"
first_part_of_path = '/nobackup/scgf/myExperiments/threeAreas/prod/prt/prt41/rt0.5/visc_2_1e25/vis1e25_mR_05/'  # remember the slash (/) at the end

#dir = first_part_of_path+'size'+str(dir_label)        # res200
dir = first_part_of_path


def read_calculate_plot(filename,scale_factor,res):
    """
    read the csv file, plot
    """
    variable_hue = "x velocity"
    myExp_all = pd.read_csv(filename, header=0)

    mask = myExp_all['y coord'] < 1    # create a mask: use these values to identify which rows to keep
    myExp1 = myExp_all[mask].copy()  # keep only the rows found by mask

    mask = myExp1['x coord'] < 1    # create a mask: use these values to identify which rows to keep
    myExp = myExp1[mask].copy()  # keep only the rows found by mask

    mask = myExp['x velocity'] == 0    # create a mask: where the horizontal velocity is 0
    x_vel_zero = myExp[mask].copy()  # keep only the rows found by mask

    if plot_figure:
        ssize = scale_factor/70.0 #scaled size for markers. 70 had a good size so 70/70 = 1 should be good
        # sns.scatterplot(data=myExp,x="x coord",y="y coord",hue=variable_hue,linewidth=0,alpha=0.8,marker="h",size=ssize,palette='Paired').set_aspect('equal') #legend='full'
        sns.scatterplot(data=myExp,x="x coord",y="y coord",hue=variable_hue,linewidth=0,alpha=0.8,marker="h",size=ssize).set_aspect('equal') #legend='full'
        # highlight the points where the x velocity is zero in yellow
        sns.scatterplot(data=x_vel_zero,x="x coord",y="y coord",color='yellow',linewidth=0,alpha=0.8,marker="h",size=ssize).set_aspect('equal') #legend='full'
        #   vmin=0, vmax=6   set ranges for broken bonds values: 0 min, 6 max
        # upper left = the point that I am setting wiht the second argument
        # plt.legend(loc='upper left',bbox_to_anchor=(0,-0.5),fancybox=True, ncol=20,prop={'size': 5.5})  
        plt.legend(loc='upper left',bbox_to_anchor=(1.05,0.9))  

        plt.xlabel('x')
        plt.ylabel('y')
    return variable_hue


# get the time step used in the simulation (do it once, from the first simulation)
input_tstep = float(getTimeStep(dir+"/input.txt")) 

# fig, axs = plt.subplots(nrows=nrow, ncols=ncol)#,constrained_layout=True)
# all_axes=axs.reshape(-1) 

timestep_number = int(re.findall(r'\d+',filename)[0])   # regex to find numbers in string. Then convert to float. Will be used to get "time" once multiplied by timestep.

print("\n",dir)
""" loop through directories"""
os.chdir(dir)

# scale_factor = float(dir_label)  # factor for scaling the axes
scale_factor = float(1)  # factor for scaling the axes



myfile = Path(os.getcwd()+'/'+filename)  # build file name including path
if myfile.is_file():
    variable_hue = read_calculate_plot(filename,scale_factor,res)
    

#  some options for plotting
if plot_figure:
    plt.title("t = "+str('{:.1e}'.format(input_tstep*timestep_number))+", "+str(variable_hue)) 
    print("---filename: ",filename,", timestep_number:",timestep_number,", input timestep: ",input_tstep,", input_tstep*timestep_number: ",input_tstep*timestep_number)
    # LEGEND:
    #box = all_axes[-1].get_position()    # get the position of the plot so I can add the legend to it 
    # upper left = the point that I am setting wiht the second argument
    #all_axes[-1].legend(loc='center left',bbox_to_anchor=(1.1,0.5),fancybox=True, ncol=1)   # legend for the vertical line plot
    # plt.tight_layout()
    # fig_name = first_part_of_path.split('/')[-2]+"_py_"+dir_label+"_bb_"+str(timestep_number)+".png" # join together all elements of the list of sizes (directories)
    # first_part_of_path -> take the second to last part (parts are separated by "/"). py because image is made with python. bb = broken bonds
    # e.g. ssx_rad200_py_060_bb_6900
    os.chdir('..')  # back to the original folder
    
    #plt.savefig(fig_name, dpi=150)#,transparent=True)
    plt.show()


# #fig.suptitle("$\sigma$ = "+sigma_str.replace("/sigma_","").replace("_",".")+", tstep = " + time_str.split("_")[1]) # get the part of the path that is after "myExperiments/"
# fig.suptitle("Fluid Pressure") # get the part of the path that is after "myExperiments/"

# # upper left = the point that I am setting wiht the second argument
# #ax2.legend(loc='center left',bbox_to_anchor=(1.2,0.5),fancybox=True, ncol=1)   # legend for the vertical line plot
# ax2.legend(fancybox=True, ncol=1)   # legend for the vertical line plot
# # save:
# #plt.savefig("gaussScale50-100-200_diff_hrz-vrt.png", dpi=600,transparent=True)
# #plt.tight_layout()
# plt.show()
