# compares results from different sims with different scales (gaussTime (200m), scale050 (50m) etc) - pressure input with gaussian distribution 
import pandas as pd
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import glob

# ---------------------------------
# - to be run from myExperiments -
# ---------------------------------

# some variables for the subplots:
tot_files = 9  # total number of subplots is (tot_files-1)*2. Each one corresponds to a time step
nrow = tot_files-1; ncol = 2;  # for each file: 1 horizontal, 1 vertical profile
fig, axs = plt.subplots(nrows=nrow, ncols=ncol)

dirList = ['/nobackup/scgf/myExperiments/gaussScale/scale050/sigma_3_0/sigma_3_0_gaussTime05/tstep04_5_1e5','/nobackup/scgf/myExperiments/gaussTime/sigma_3_0/sigma_3_0_gaussTime05/tstep04_5_1e5']

# initialise stuff
max_P = 0.0; min_P = 0.0

#for myDir in dirList[1:4]:
for myDir in dirList:
    # loop through directories
    os.chdir(myDir)
    print("--> Entering " + str(os.getcwd()))
    file_list = sorted(glob.glob("my_experiment00*.csv"))
    #print(file_list)

    myExp0 = pd.read_csv(file_list[3], header=0)  # as a reference: open a later file to get xmax,ymax,x coordinates
    xmax = myExp0.loc[myExp0['Pressure'].idxmax(), 'x coord']
    ymax = myExp0.loc[myExp0['Pressure'].idxmax(), 'y coord']

    # find the x coordinates that corresponds to the max pressure
    x_array = np.array(myExp0.loc[myExp0.apply(lambda x: math.isclose(x['y coord'],ymax,rel_tol=1e-3),axis=1),'x coord'])

    # get the y coordinates of all x in range of tolerance
    y_array = np.array(myExp0.loc[myExp0.apply(lambda x: math.isclose(x['x coord'],xmax,rel_tol=1e-3),axis=1),'y coord'])

    for i,file in enumerate(file_list[0:tot_files]): 

        if i == 0 or i == 1:
            continue
        #print(file)
        # go through files, one by one
        myExp = pd.read_csv(file, header=0)

        max_P_temp = max(myExp['Pressure'])  # I want the maximum pressure. Save here temporarily.       
        if max_P_temp > max_P:
            max_P = max_P_temp    # if it's greater than what I had before, switch

        min_P_temp = min(myExp['Pressure'])  # same for minimum
        if min_P_temp > min_P:
            min_P = min_P_temp

        # extract the pressure arrays
        pressure_array_x = np.array(myExp.loc[myExp.apply(lambda x: math.isclose(x['y coord'],ymax,rel_tol=1e-3),axis=1),'Pressure'])
        pressure_array_y = np.array(myExp.loc[myExp.apply(lambda x: math.isclose(x['x coord'],xmax,rel_tol=1e-3),axis=1),'Pressure'])
        x = str(myDir).split("myExperiments/")  # split between before and after myExperiments/
        labelName = x[1]  # take the second part of the string
        all_axes=axs.reshape(-1)
        all_axes[2*i-2].plot(x_array, pressure_array_x,label=labelName)
        all_axes[2*i-1].plot(y_array, pressure_array_y,label=labelName)
    os.chdir("..")

# LEGEND:
# get the position of the middle plot so I can add the legend to it (all_axes[1].legend)
middle_subplot = int(tot_files/2)
#box = all_axes[middle_subplot].get_position()

box = all_axes[-1].get_position()

#fig.suptitle("Profiles along the point of Maximum Pressure")

# upper left = the point that I am setting wiht the second argument
#ax3.legend(loc='upper left',bbox_to_anchor=(0,-0.1),fancybox=True, ncol=8)

all_axes[-2].legend(loc='center left',bbox_to_anchor=(0,-0.5),fancybox=True, ncol=10) 
all_axes[middle_subplot].set_ylabel("Fluid Pressure")   # the middle plot (tot_files/2)
#all_axes[-1].set_ylabel("Fluid Pressure")   # the middle plot (tot_files/2)
all_axes[0].set_title("Horizontal Profile")
all_axes[0].set_title("Vertical Profile")


for i,ax in enumerate(all_axes):
    # zoom in. Limits are max location +- 0.05
    if i%2 == 0: # even => horizontal
        all_axes[i].set_xlim([xmax-0.01,xmax+0.01])
    else:  # vertical profile
        all_axes[i].set_xlim([ymax-0.01,ymax+0.01])
    all_axes[i].set_ylim([min_P,max_P])  

#plt.tight_layout()
os.chdir('..')
#plt.savefig(parent_directory+"_diffusion_Pprofile_smallinterval.png",dpi=600)
plt.show()
"""
    

# -- end directory loop --

ax1.set_ylabel("Fluid Pressure")
ax1.set_title("Horizontal Profile")
ax2.set_title("Vertical Profile")

# zoom in. Limits are max location +- 0.05
ax1.set_xlim([xmax-0.05,xmax+0.05])
ax2.set_xlim([ymax-0.05,ymax+0.05])
# LEGEND:
# get the position of the 3rd plot so I can add the legend to it (ax3.legend)
box = ax2.get_position()

#fig.suptitle("Profiles along the point of Maximum Pressure")

# upper left = the point that I am setting wiht the second argument
#ax3.legend(loc='upper left',bbox_to_anchor=(0,-0.1),fancybox=True, ncol=8)
ax2.legend(loc='center left',bbox_to_anchor=(1,0.5),fancybox=True, ncol=1) 
#plt.tight_layout()
plt.show()

"""
