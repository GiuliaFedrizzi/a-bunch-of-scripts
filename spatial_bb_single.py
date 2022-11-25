""" 
builds matrix:



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

from useful_functions import * 
from useful_functions_moments import *

plot_figure = 1
#dir_labels = ['020','030','040']#,'050','060','070','080','090','100']   
dir_label = '020'#,'080','090','100']   
#dir_labels = ['090','100']   
filename = "my_experiment07000.csv"


first_part_of_path = '/nobackup/scgf/myExperiments/gaussScaleFixFrac2/sxx_rad200'
dir = first_part_of_path+'/size'+str(dir_label)        # res200


def read_calculate_plot(filename,scale_factor,res):
    """
    read the csv file, call build_cov_matrix, calculate eigenvalues and vectors, plot
    """
    myExp = pd.read_csv(filename, header=0)

    mask = myExp['Broken Bonds'] > 0    # create a mask: find the true value of each row's 'xcoord' column being greater than 0, then using those truth values to identify which rows to keep
    bb_df = myExp[mask].copy()  # keep only the rows found by mask

    bb_df.reset_index(drop=True, inplace=True)

    # factors to shift the domain so they are all consistent. Reference is size = 100
    x_shift = 50-(scale_factor/2)   # 50 is the position of the centre in the simulation w size = 100
    y_shift = 100-(scale_factor)   
    
    bb_df['xcoord100'] = bb_df['x coord']*scale_factor + x_shift
    bb_df['ycoord100'] = bb_df['y coord']*scale_factor + y_shift

    tot_bb = bb_df['Broken Bonds'].sum()   # 0th order moment
    
    if len(bb_df)==0:
        print("No broken bonds.")
        return

    cov_matrix,com_x,com_y = build_cov_matrix(bb_df,tot_bb)

    #print(cov_matrix)

    w, v = np.linalg.eig(cov_matrix)  # eigenvalues and eigenvectors
    print("eigenvalues: " + str(w)+ ", \neigenvectors: "+ str(v))

    theta_rad = np.pi/2 - np.arctan2(2*cov_matrix[0][1],(cov_matrix[0][0]-cov_matrix[1][1]))/2  # orientation in rad   2*b,a-c
    theta_deg = theta_rad % 180 - 90         # orientation in degrees
    # if (a-c)*math.cos(2*theta)+b*math.sin(2*theta) > 0: # it's maximising the second moment. wrong theta.
    #     theta = theta + math.pi

    print("theta (rad) = ",theta_rad," theta (deg) = ",theta_deg)
    # TO DO: weight sum by n of bb

    if plot_figure:
        #print("dirnum: ",dirnum)
        #plt.sca(all_axes[int(dirnum)])   # set the current active subplot        
        # sns.scatterplot(data=bb_df,x="xcoord100",y="ycoord100",hue="Broken Bonds",linewidth=0,alpha=0.8,marker="h",size=0.6).set_aspect('equal')
        ssize = scale_factor/70.0 #scaled size for markers. 70 had a good size so 70/70 = 1 should be good
        sns.scatterplot(data=bb_df,x="xcoord100",y="ycoord100",hue="Broken Bonds",linewidth=0,alpha=0.8,marker="h",size=ssize,vmin=0, vmax=6).set_aspect('equal')
        #   vmin=0, vmax=6   set ranges for broken bonds values: 0 min, 6 max
        plt.plot(com_x,com_y,'.',markersize=10)

        x1 = com_x + math.cos(theta_rad)*0.5*min(w)
        y1 = com_y - math.sin(theta_rad)*0.5*min(w)
        x2 = com_x - math.sin(theta_rad)*0.5*max(w)
        y2 = com_y - math.cos(theta_rad)*0.5*max(w)
        plt.plot((com_x, x1), (com_y, y1), '-r', linewidth=2.5)
        plt.plot((com_x, x2), (com_y, y2), '-r', linewidth=2.5)
        plt.legend([],[], frameon=False)
        plt.title("Scale = "+str(scale_factor)+" res = "+res)
        plt.axhline(y=100, color='k', linestyle='-')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim([40,60])
        plt.ylim([80,100])
        #y1 = com_y - math.sin(theta_rad)*0.5*min(w)
        # plt.arrow(com_x,com_y,v[0][0]/4,v[0][1]/4, head_width=0.03, head_length=0.03)  # first eigenvector: arrow from 0,0 (origin) to the coordinates of 1st eigv
        # plt.arrow(com_x,com_y,v[1][0]/4,v[1][1]/4, head_width=0.03, head_length=0.03)
    

# get the time step used in the simulation (do it once, from the first simulation)
input_tstep = float(getTimeStep(dir+"/input.txt")) 

# fig, axs = plt.subplots(nrows=nrow, ncols=ncol)#,constrained_layout=True)
# all_axes=axs.reshape(-1) 

timestep_number = int(re.findall(r'\d+',filename)[0])   # regex to find numbers in string. Then convert to float. Will be used to get "time" once multiplied by timestep.

print("\n",dir)
""" loop through directories"""
os.chdir(dir)

scale_factor = float(dir_label)  # factor for scaling the axes

res = get_resolution(dir)

myfile = Path(os.getcwd()+'/'+filename)  # build file name including path
if myfile.is_file():
    read_calculate_plot(filename,scale_factor,res)
    

#  some options for plotting
if plot_figure:
    plt.title("t = "+str('{:.1e}'.format(input_tstep*timestep_number))) 
    print("---filename: ",filename,", timestep_number:",timestep_number,", input timestep: ",input_tstep,", input_tstep*timestep_number: ",input_tstep*timestep_number)
    # LEGEND:
    #box = all_axes[-1].get_position()    # get the position of the plot so I can add the legend to it 
    # upper left = the point that I am setting wiht the second argument
    #all_axes[-1].legend(loc='center left',bbox_to_anchor=(1.1,0.5),fancybox=True, ncol=1)   # legend for the vertical line plot
    plt.tight_layout()
    fig_name = "ssx_rad200_py_"+dir_label+"_bb_"+str(timestep_number)+".png" # join together all elements of the list of sizes (directories)
    # ssx_rad200_py_s060_bb_6900
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
