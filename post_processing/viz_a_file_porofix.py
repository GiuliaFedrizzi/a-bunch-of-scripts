""" 
builds matrix from csv, plots it
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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
filename = "my_experiment00300.csv"
first_part_of_path = '/nobackup/scgf/myExperiments/threeAreas/prod/prt/singleInjection/porosity_fix/pf01'  # remember the slash (/) at the end

#dir = first_part_of_path+'size'+str(dir_label)        # res200
dir = first_part_of_path


def read_calculate_plot(filename,scale_factor,res,var_plot):
    """
    read the csv file, plot
    """
    variable_hue = var_plot 

    myExp_all = pd.read_csv(filename, header=0)
    # myExp_all["x coord"] = myExp_all["x coord"]-0.0045
    # myExp_all["y coord"] = myExp_all["y coord"]-0.006
    

    myExpFluid_all = pd.read_csv("fluidgrid00300.csv", header=0)
    myExpFluid_all["x"] = myExpFluid_all["x"]/100
    myExpFluid_all["y"] = myExpFluid_all["y"]/100

    myExp = myExp_all[(myExp_all['y coord'] >= 0.48) & 
        (myExp_all['y coord'] <= 0.52) &
        (myExp_all['x coord'] >= 0.48) &
        (myExp_all['x coord'] <= 0.52)
        ]

    myExpFluid = myExpFluid_all[(myExpFluid_all['y'] >= 0.48) & 
        (myExpFluid_all['y'] <= 0.52) &
        (myExpFluid_all['x'] >= 0.48) &
        (myExpFluid_all['x'] <= 0.52)
        ]
    print(f"myExpFluid max {max(myExpFluid['Porosity'])}, min {min(myExpFluid['Porosity'])}")
    if plot_figure:
        plt.figure()
        ssize = 30 #scaled size for markers. Worked for plt.show 
        # ssize = 4 #scaled size for markers
        
        scatter = sns.scatterplot(data=myExp,x="x coord",y="y coord",hue=variable_hue,
            linewidth=0,alpha=0.8,marker="h",s=ssize,palette="cividis",legend=False).set_aspect('equal') #legend='full'
        
        scatter = sns.scatterplot(data=myExpFluid,x="x",y="y",hue=variable_hue,
            linewidth=0,alpha=0.8,marker="s",s=ssize*2,palette="cividis",legend=False).set_aspect('equal') #legend='full'
        
        for i in range(48,53):
            plt.axvline(x = i/100,color = 'k',alpha=0.5)
            plt.axhline(y = i/100,color = 'k',alpha=0.5)

        # upper left = the point that I am setting wiht the second argument
        # plt.legend(loc='upper left',bbox_to_anchor=(0,-0.5),fancybox=True, ncol=20,prop={'size': 5.5})  

        ax = plt.gca()
        # plt.legend(loc='upper left',bbox_to_anchor=(1.05,0.9))  
        plt.xlabel('x')
        plt.ylabel('y')
        
        # # plt.show()
        # plt.savefig('stress_'+stress_type+'_'+ '{:.0e}'.format(threshold)+'.png',dpi=400)
        # plt.clf()
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
    variable_hue = read_calculate_plot(filename,scale_factor,res,"Porosity")
    

#  some options for plotting
if plot_figure:
    plt.title("t = "+str('{:.1e}'.format(input_tstep*timestep_number))+", "+str(variable_hue)) 
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
