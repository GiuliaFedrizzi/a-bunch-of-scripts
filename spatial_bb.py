""" 
builds matrix:



"""

# I use this ^ to run python in VS code in interactive mode
import pandas as pd
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy import interpolate
import os
import glob
import seaborn as sns
from pathlib import Path


plot_figure = 1
#fig, axs = plt.subplots(nrows=1, ncols=1)
dir_labels = ['020','030','040','050']   # must be even
# dir_labels = ['040','050','075','100','150']
resolution = 400

# some variables for the subplots:
tot_files = len(dir_labels)  # is going to be the total number of subplots (#subplots=tot_files-1)
ncol = int((tot_files)); nrow = 1;
fig, axs = plt.subplots(nrows=nrow, ncols=ncol)
all_axes=axs.reshape(-1) 

dir_list = []
origin = np.array([[0, 0],[0, 0]]) # origin point

for i in dir_labels:
    dir_list.append('/nobackup/scgf/myExperiments/gaussScaleFixFrac/fixedfrac100/size'+str(i))
    
#fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2)
first_file = 'my_experiment00010.csv'
df_x = pd.DataFrame()
df_y = pd.DataFrame()

def getTimeStep(inputFile):
    with open(inputFile) as iFile:
        for num, line in enumerate(iFile,1):
            if "Tstep" in line:
                input_tstep = line.split(" ")[1]  # split before and after space, take the second word (value of timestep)
                return input_tstep  
def getViscosity(inputFile):
    with open(inputFile) as iFile:
        for num, line in enumerate(iFile,1):
            if "mu_f" in line:
                input_mu_f = line.split(" ")[1]  # split before and after space, take the second word (value of timestep)
                return input_mu_f  

dirnum = 0 # counter for directories
for dir in dir_list:
    print(dir)
    """ loop through directories"""
    os.chdir(dir)
    
    dir_label = dir_labels[dirnum]   # get the label that corresponds to the directory
    
    scale_factor = float(dir_label)/200.0# factor for scaling the axes

    # open the first file after melt addition to find the max P
    myExp = pd.read_csv(first_file, header=0)
    # xmax = myExp.loc[myExp['Pressure'].idxmax(), 'x coord']
    # ymax = myExp.loc[myExp['Pressure'].idxmax(), 'y coord']
    max_id = myExp['Pressure'].idxmax() 
    
    filename = "my_experiment00250.csv"
    #for i,filename in enumerate(sorted(glob.glob("my_experiment*"))[2:36:5]): #[beg:end:step]
    myfile = Path(os.getcwd()+'/'+filename)  # build file name including path
    if myfile.is_file():
        cov_matrix = np.empty([2,2])  # initialise covariant matrix. 2x2
        myExp = pd.read_csv(filename, header=0)
        #print(len(myExp))
        bb_df =  myExp[myExp['Broken Bonds']>0]   # select only those with at least one broken bond
        #print(len(bb_df))
        
        tot_bb = bb_df['Broken Bonds'].sum()   # 0th order moment
        print("tot bb = ",tot_bb)
        input_tstep = float(getTimeStep("input.txt"))
        if len(bb_df)==0:
            print("No broken bonds.")
            continue
        # fnx = lambda row: row['x coord'] - xmax
        # colx = bb_df.apply(fnx,axis=1)
        # #bb_df['x coord shifted'] = bb_df['x coord'] - xmax
        # bb_df = bb_df.assign(x_coord_shifted=colx.values)

        # fny = lambda row: row['y coord'] - ymax
        # coly = bb_df.apply(fny,axis=1)
        # #bb_df['x coord shifted'] = bb_df['x coord'] - xmax
        # bb_df = bb_df.assign(y_coord_shifted=coly.values)



        # 1st order moment: position of centre of mass (COM)
        com_x = (bb_df['x coord']*bb_df['Broken Bonds']).sum()/tot_bb  # multiplies row by row, then sums everything
        com_y = (bb_df['y coord']*bb_df['Broken Bonds']).sum()/tot_bb  # multiplies row by row, then sums everything
        
        bb_df['x_coord_shifted'] = bb_df['x coord']-com_x
        bb_df['y_coord_shifted'] = bb_df['y coord']-com_y

        # # get some info about the file
        # input_viscosity = float(getViscosity("input.txt"))
        # file_num = float(filename.split("experiment")[1].split(".")[0])  # first take the part after "experiment", then the one before the "."

        # # name of the line in the plot
        # labelName = "t/$\mu$=" + str('{:.1e}'.format(input_tstep*file_num/input_viscosity))

        
        a = ((bb_df['x coord']-com_x)**2*bb_df['Broken Bonds']).sum()#/tot_bb  # multiplies tow by row, then sums everything
        b = ((bb_df['x coord']-com_x)*(bb_df['y coord']-com_y)*bb_df['Broken Bonds']).sum()#/tot_bb  # multiplies tow by row, then sums everything
        c = ((bb_df['y coord']-com_y)**2*bb_df['Broken Bonds']).sum()#/tot_bb  # multiplies tow by row, then sums everything
        
        # a = ((bb_df['x coord']-com_x)**2).sum()/tot_bb  # multiplies tow by row, then sums everything
        # b = ((bb_df['x coord']-com_x)*(bb_df['y coord']-com_y)).sum()/tot_bb  # multiplies tow by row, then sums everything
        # c = ((bb_df['y coord']-com_y)**2).sum()/tot_bb  # multiplies tow by row, then sums everything
        
        cov_matrix[0][0]=a
        cov_matrix[0][1]=b
        cov_matrix[1][0]=b
        cov_matrix[1][1]=c
        print(cov_matrix)

        w, v = np.linalg.eig(cov_matrix)  # eigenvalues and eigenvectors
        print("eigenvalues: " + str(w)+ ", \neigenvectors: "+ str(v))

        # true_v = 2*v        
        # # true_v=np.empty([2,1])
        # # # scale the eigenvectors so they have true units
        # # true_v[0] = 2*math.sqrt(v[0]/tot_bb)
        # # true_v[1] = 2*math.sqrt(v[1]/tot_bb)
        # print("true v = ",true_v)
        #print("v = ",v)

        theta_rad = np.arctan2(2*b,(a-c))/2  # orientation in rad
        theta_deg = theta_rad % 180 - 90         # orientation in degrees
        # if (a-c)*math.cos(2*theta)+b*math.sin(2*theta) > 0: # it's maximising the second moment. wrong theta.
        #     theta = theta + math.pi

        print("theta (rad) = ",theta_rad," theta (deg) = ",theta_deg)
        # TO DO: weight sum by n of bb

        if plot_figure:
            print("figure ",dirnum)
            plt.sca(all_axes[dirnum])           
            sns.scatterplot(data=bb_df,x="x coord",y="y coord",hue="Broken Bonds",linewidth=0,alpha=0.8,marker="h",size=0.6).set_aspect('equal')
            # plt.arrow(0,0,v[0][0]/4,v[0][1]/4, head_width=0.03, head_length=0.03)  # first eigenvector: arrow from 0,0 (origin) to the coordinates of 1st eigv
            # plt.arrow(0,0,v[1][0]/4,v[1][1]/4, head_width=0.03, head_length=0.03)
            plt.plot(com_x,com_y,'.',markersize=10)

            x1 = com_x + math.cos(theta_rad)*0.5*min(w)
            y1 = com_y - math.sin(theta_rad)*0.5*min(w)
            x2 = com_x - math.sin(theta_rad)*0.5*max(w)
            y2 = com_y - math.cos(theta_rad)*0.5*max(w)
            plt.plot((com_x, x1), (com_y, y1), '-r', linewidth=2.5)
            plt.plot((com_x, x2), (com_y, y2), '-r', linewidth=2.5)
            #y1 = com_y - math.sin(theta_rad)*0.5*min(w)
            # plt.arrow(com_x,com_y,v[0][0]/4,v[0][1]/4, head_width=0.03, head_length=0.03)  # first eigenvector: arrow from 0,0 (origin) to the coordinates of 1st eigv
            # plt.arrow(com_x,com_y,v[1][0]/4,v[1][1]/4, head_width=0.03, head_length=0.03)
            
        dirnum+=1 
        # end of for loop through files
if plot_figure:
    plt.show()
#g_x = sns.lineplot(data=df_x ,x="x",y="Stress (MPa)",ax=ax1,hue='time',style='scale',alpha=0.5)
#g_y = sns.lineplot(data=df_y ,x="y",y="Stress (MPa)",ax=ax2,hue='time',style='scale',alpha=0.5)
# g_x = sns.lineplot(data=df_x ,x="x",y="Fluid Pressure (MPa)",ax=ax1,hue='time',style='scale',alpha=0.5)
# g_y = sns.lineplot(data=df_y ,x="y",y="Fluid Pressure (MPa)",ax=ax2,hue='time',style='scale',alpha=0.5)
# #ax1.set_ylabel("Stress")
# ax1.set_title("Horizontal Profile")
# ax1.set_xlim([-0.06,+0.06])  # zoom in. Limits are max location +- 0.06
# ax2.set_xlim([-0.06,+0.06])  # zoom in. Limits are max location +- 0.06
# #ax1.set_ylim([1e8,1.3e8])  # zoom in.
# #ax2.set_ylim([1e8,1.3e8])  # zoom in. 
# #ax2.plot(y_array, pressure_array_y,plotStyle,label=labelName)
# #ax2.legend()
# ax2.set_title("Vertical Profile")

# g_x.legend_.remove()
# #fig.suptitle("$\sigma$ = "+sigma_str.replace("/sigma_","").replace("_",".")+", tstep = " + time_str.split("_")[1]) # get the part of the path that is after "myExperiments/"
# fig.suptitle("Fluid Pressure") # get the part of the path that is after "myExperiments/"

# # LEGEND:
# # get the position of the plot so I can add the legend to it 
# box = ax2.get_position()

# # upper left = the point that I am setting wiht the second argument
# #ax2.legend(loc='center left',bbox_to_anchor=(1.2,0.5),fancybox=True, ncol=1)   # legend for the vertical line plot
# ax2.legend(fancybox=True, ncol=1)   # legend for the vertical line plot
# # save:
# #os.chdir('/nobackup/scgf/myExperiments/gaussScale')
# #plt.savefig("gaussScale50-100-200_diff_hrz-vrt.png", dpi=600,transparent=True)
# #plt.tight_layout()
# plt.show()
