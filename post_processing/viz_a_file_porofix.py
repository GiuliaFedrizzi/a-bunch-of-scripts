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
import matplotlib as mpl

from useful_functions import * 
# from useful_functions_moments import *

# options to set:
plot_figure = 1
res = 200
filename = "my_experiment"
filenum = "00000"
first_part_of_path = '/nobackup/scgf/myExperiments/threeAreas/prod/prt/singleInjection/porosity_fix/'  
pf = 'pf01'
dir = first_part_of_path + pf


def read_calculate_plot(filenum,scale_factor,res,var_plot):
    """
    read the csv file, plot
    """
    variable_hue = var_plot 

    myExp_all = pd.read_csv("my_experiment"+filenum+".csv", header=0)
    # myExp_all["x coord"] = myExp_all["x coord"]-0.0045
    # myExp_all["y coord"] = myExp_all["y coord"]+0.0005
    

    myExpFluid_all = pd.read_csv("fluidgrid"+filenum+".csv", header=0)
    myExpFluid_all["x"] = myExpFluid_all["x"]/100
    myExpFluid_all["y"] = myExpFluid_all["y"]/100
    print(f' {myExpFluid_all.head()}')

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
        ssize = 50 #scaled size for markers. Worked for plt.show 
        col_palette = "afmhot" #"gnuplot"

        max_value = myExpFluid[variable_hue].max()#*1.1
        min_value = myExpFluid[variable_hue].min()#*1.05
        norm = mpl.colors.Normalize(vmin=min_value, vmax=max_value)
        
        scatter = sns.scatterplot(data=myExp,x="x coord",y="y coord",hue=variable_hue,
            linewidth=0,alpha=0.8,marker="o",s=ssize,hue_norm=(min_value,max_value),palette=col_palette,legend=False).set_aspect('equal') #legend='full'
        
        scatter = sns.scatterplot(data=myExpFluid,x="x",y="y",hue=variable_hue,
            linewidth=0,alpha=0.8,marker="s",s=ssize*2,hue_norm=(min_value,max_value),palette=col_palette,legend=False).set_aspect('equal') #legend='full'
        # Adding a color bar
        norm = plt.Normalize(vmin=min_value, vmax=max_value)
        sm = plt.cm.ScalarMappable(cmap=col_palette, norm=norm)
        sm.set_array([])  # Only needed for the colorbar to display correctly
        plt.colorbar(sm, label=variable_hue)  # Adding color bar with label

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
        # plt.savefig('porofix_'+pf+'_'+ 't'+filenum+'.png',dpi=400)
        # plt.clf()
    return variable_hue


# get the time step used in the simulation (do it once, from the first simulation)
input_tstep = float(getTimeStep(dir+"/input.txt")) 

# fig, axs = plt.subplots(nrows=nrow, ncols=ncol)#,constrained_layout=True)
# all_axes=axs.reshape(-1) 

timestep_number = int(filenum)
#= int(re.findall(r'\d+',filename)[0])   # regex to find numbers in string. Then convert to float. Will be used to get "time" once multiplied by timestep.

print("\n",dir)
""" loop through directories"""
os.chdir(dir)

scale_factor = float(1)  # factor for scaling the axes

myfile = Path(os.getcwd()+'/my_experiment'+filenum+".csv")  # build file name including path
if myfile.is_file():
    variable_hue = read_calculate_plot(filenum,scale_factor,res,"Porosity")
    

#  some options for plotting
if plot_figure:
    # plt.title(pf+", t = "+str('{:.1e}'.format(input_tstep*timestep_number))+", "+str(variable_hue)) 
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
