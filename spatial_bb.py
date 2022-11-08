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

from useful_functions import * 

plot_figure = 1
#dir_labels = ['020','030','040','050','060','070','080','090','100']   
dir_labels = ['090','100']   
resolution = 200

# some variables for the subplots:
tot_files = len(dir_labels)  # is going to be the total number of subplots (#subplots=tot_files-1)
nrow = 1;
ncol = int((tot_files)/nrow);
fig, axs = plt.subplots(nrows=nrow, ncols=ncol)
all_axes=axs.reshape(-1) 

dir_list = []

for i in dir_labels:
    dir_list.append('/nobackup/scgf/myExperiments/gaussScaleFixFrac2/radiusSq200/size'+str(i))


def build_cov_matrix(bb_df,tot_bb):
    """ Calculate the first order moment = centre of mass (com_x and com_y are the coordinates),
    build the covariant matrix from the broken bonds dataframe
     """

    cov_matrix = np.empty([2,2])  # initialise covariant matrix. 2x2
    
    # 1st order moment: position of centre of mass (COM)
    com_x = (bb_df['xcoord100']*bb_df['Broken Bonds']).sum()/tot_bb  # multiplies row by row, then sums everything
    com_y = (bb_df['ycoord100']*bb_df['Broken Bonds']).sum()/tot_bb  # multiplies row by row, then sums everything
    
    #bb_df['x_coord_shifted'] = bb_df['xcoord100']-com_x
    #bb_df['y_coord_shifted'] = bb_df['ycoord100']-com_y
    a = ((bb_df['xcoord100']-com_x)**2*bb_df['Broken Bonds']).sum()/tot_bb  # multiplies tow by row, then sums everything
    b = ((bb_df['xcoord100']-com_x)*(bb_df['ycoord100']-com_y)*bb_df['Broken Bonds']).sum()/tot_bb  # multiplies tow by row, then sums everything
    c = ((bb_df['ycoord100']-com_y)**2*bb_df['Broken Bonds']).sum()/tot_bb  # multiplies tow by row, then sums everything
    
    # a = ((bb_df['x coord']-com_x)**2).sum()/tot_bb  # multiplies tow by row, then sums everything
    # b = ((bb_df['x coord']-com_x)*(bb_df['y coord']-com_y)).sum()/tot_bb  # multiplies tow by row, then sums everything
    # c = ((bb_df['y coord']-com_y)**2).sum()/tot_bb  # multiplies tow by row, then sums everything
    
    cov_matrix[0][0]=a
    cov_matrix[0][1]=b
    cov_matrix[1][0]=b
    cov_matrix[1][1]=c

    return cov_matrix,com_x,com_y

def read_calculate_plot(filename,scale_factor):
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
    print("scale factor: ",scale_factor)
    bb_df['xcoord100'] = bb_df['x coord']*scale_factor + x_shift
    bb_df['ycoord100'] = bb_df['y coord']*scale_factor + y_shift

    tot_bb = bb_df['Broken Bonds'].sum()   # 0th order moment
    
    
    if len(bb_df)==0:
        print("No broken bonds.")
        return

    cov_matrix,com_x,com_y = build_cov_matrix(bb_df,tot_bb)

    print(cov_matrix)

    w, v = np.linalg.eig(cov_matrix)  # eigenvalues and eigenvectors
    print("eigenvalues: " + str(w)+ ", \neigenvectors: "+ str(v))

    theta_rad = np.pi/2 - np.arctan2(2*cov_matrix[0][1],(cov_matrix[0][0]-cov_matrix[1][1]))/2  # orientation in rad   2*b,a-c
    theta_deg = theta_rad % 180 - 90         # orientation in degrees
    # if (a-c)*math.cos(2*theta)+b*math.sin(2*theta) > 0: # it's maximising the second moment. wrong theta.
    #     theta = theta + math.pi

    print("theta (rad) = ",theta_rad," theta (deg) = ",theta_deg)
    # TO DO: weight sum by n of bb

    if plot_figure:
        plt.sca(all_axes[dirnum])   # set the current active subplot        
        # sns.scatterplot(data=bb_df,x="xcoord100",y="ycoord100",hue="Broken Bonds",linewidth=0,alpha=0.8,marker="h",size=0.6).set_aspect('equal')
        sns.scatterplot(data=bb_df,x="xcoord100",y="ycoord100",hue="Broken Bonds",linewidth=0,alpha=0.8,marker="h").set_aspect('equal')
        
        plt.plot(com_x,com_y,'.',markersize=10)

        x1 = com_x + math.cos(theta_rad)*0.5*min(w)
        y1 = com_y - math.sin(theta_rad)*0.5*min(w)
        x2 = com_x - math.sin(theta_rad)*0.5*max(w)
        y2 = com_y - math.cos(theta_rad)*0.5*max(w)
        plt.plot((com_x, x1), (com_y, y1), '-r', linewidth=2.5)
        plt.plot((com_x, x2), (com_y, y2), '-r', linewidth=2.5)
        plt.legend([],[], frameon=False)
        #y1 = com_y - math.sin(theta_rad)*0.5*min(w)
        # plt.arrow(com_x,com_y,v[0][0]/4,v[0][1]/4, head_width=0.03, head_length=0.03)  # first eigenvector: arrow from 0,0 (origin) to the coordinates of 1st eigv
        # plt.arrow(com_x,com_y,v[1][0]/4,v[1][1]/4, head_width=0.03, head_length=0.03)
        

dirnum = 0 # counter for directories
input_tstep = float(getTimeStep(dir_list[0]+"/input.txt"))

for dir in dir_list:
    print("\n",dir)
    """ loop through directories"""
    os.chdir(dir)
    
    dir_label = dir_labels[dirnum]   # get the label that corresponds to the directory
    
    scale_factor = float(dir_label)  # factor for scaling the axes

    filename = "my_experiment00050.csv"
    #for i,filename in enumerate(sorted(glob.glob("my_experiment*"))[2:36:5]): #[beg:end:step]  # loop through files
    myfile = Path(os.getcwd()+'/'+filename)  # build file name including path
    if myfile.is_file():
        read_calculate_plot(filename,scale_factor)
        
    dirnum+=1  # advance the counter, even if no files were found
    

#  some options for plotting
if plot_figure:
    fig.suptitle("t = "+str(input_tstep))

    # LEGEND:
    box = all_axes[-1].get_position()    # get the position of the plot so I can add the legend to it 

    # upper left = the point that I am setting wiht the second argument
    all_axes[-1].legend(loc='center left',bbox_to_anchor=(1.2,0.5),fancybox=True, ncol=1)   # legend for the vertical line plot

    plt.show()


# #fig.suptitle("$\sigma$ = "+sigma_str.replace("/sigma_","").replace("_",".")+", tstep = " + time_str.split("_")[1]) # get the part of the path that is after "myExperiments/"
# fig.suptitle("Fluid Pressure") # get the part of the path that is after "myExperiments/"

# # upper left = the point that I am setting wiht the second argument
# #ax2.legend(loc='center left',bbox_to_anchor=(1.2,0.5),fancybox=True, ncol=1)   # legend for the vertical line plot
# ax2.legend(fancybox=True, ncol=1)   # legend for the vertical line plot
# # save:
# #os.chdir('/nobackup/scgf/myExperiments/gaussScale')
# #plt.savefig("gaussScale50-100-200_diff_hrz-vrt.png", dpi=600,transparent=True)
# #plt.tight_layout()
# plt.show()
