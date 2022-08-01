# compares results from different sims in different directories (gaussTime) - pressure input with gaussian distribution 
import pandas as pd
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import glob

#parent_directory = "gaussTimeOnce"

# some variables for the subplots:
tot_files = 9  # is going to be the total number of subplots (#subplots=tot_files-1)
nrow = tot_files-1; ncol = 2;
fig, axs = plt.subplots(nrows=nrow, ncols=ncol)


#fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2)
#fig, ax1 = plt.subplots(nrows=1,ncols=1)

#ax3.set_xlim([xmax-0.05,xmax+0.05])
#ax4.set_xlim([ymax-0.05,ymax+0.05])
#ax3.set_ylim([0.054,0.055])    


#os.chdir(parent_directory)
dir_list_sigma = []
#for my_dir in sorted(glob.glob("tstep*")):
for my_dir in sorted(glob.glob("sigma*")):
    """ get the list of directories"""
    dir_list_sigma.append(my_dir)
print(dir_list_sigma)
max_P = 0.0
min_P = 0.0

#for myDir in dir_list_sigma[1:4]:
for myDir in dir_list_sigma[1:3]:
    # loop through directories
    os.chdir(myDir + "/" + myDir + "_gaussTime05/tstep04_4_1e4")
    print("--> Entering " + str(os.getcwd()))
    file_list = sorted(glob.glob("my_experiment0*.csv"))
    file_list = file_list[2:tot_files+2]
    print("list:" + ' '.join(file_list) + " size = " + str(len(file_list)))
    #print(file_list)

    # things that are done only once: locate the max, get the coordinates
    myExp0 = pd.read_csv(file_list[3], header=0)  # as a reference: open a later file to get xmax,ymax,x coordinates
    xmax = myExp0.loc[myExp0['Pressure'].idxmax(), 'x coord']
    ymax = myExp0.loc[myExp0['Pressure'].idxmax(), 'y coord']

    # find the x coordinates that corresponds to the max pressure
    x_array = np.array(myExp0.loc[myExp0.apply(lambda x: math.isclose(x['y coord'],ymax,rel_tol=1e-3),axis=1),'x coord'])

    # get the y coordinates of all x in range of tolerance
    y_array = np.array(myExp0.loc[myExp0.apply(lambda x: math.isclose(x['x coord'],xmax,rel_tol=1e-3),axis=1),'y coord'])


    for i,file in enumerate(file_list):
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
        x = str(os.getcwd())  # current directory
        labelName = '-'.join(x.split("/")[5:-1]) # split where there there's a slash, keep dir names from the 6th (5) to the last (-1), join these with a -
        all_axes=axs.reshape(-1)
        all_axes[2*i-2].plot(x_array, pressure_array_x,label=labelName)
        all_axes[2*i-1].plot(y_array, pressure_array_y,label=labelName)
    os.chdir("../../..")

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
