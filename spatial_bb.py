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
dir_labels = ['040','050']
# dir_labels = ['040','050','075','100','150']
resolution = 400


dir_list = []
origin = np.array([[0, 0],[0, 0]]) # origin point

for i in dir_labels:
    dir_list.append('/nobackup/scgf/myExperiments/gaussScaleFixFrac/fixedfracs3/size'+str(i))
    
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

dirnum = 0
for dir in dir_list:
    print(dir)
    """ loop through directories"""
    os.chdir(dir)
    dirnum+=1 # counter for directories
    dir_label = dir_labels[dirnum-1]   # get the label that corresponds to the directory
    
    scale_factor = float(dir_label)/200.0# factor for scaling the axes

    # open the first file after melt addition to find the max P
    myExp = pd.read_csv(first_file, header=0)
    xmax = myExp.loc[myExp['Pressure'].idxmax(), 'x coord']
    ymax = myExp.loc[myExp['Pressure'].idxmax(), 'y coord']
    max_id = myExp['Pressure'].idxmax() 
    
    filename = "my_experiment00500.csv"
    #for i,filename in enumerate(sorted(glob.glob("my_experiment*"))[2:36:5]): #[beg:end:step]
    myfile = Path(os.getcwd()+'/'+filename)  # build file name including path
    if myfile.is_file():
        cov_matrix = np.empty([4,1])
        myExp = pd.read_csv(filename, header=0)
        #print(len(myExp))
        bb_df =  myExp[myExp['Broken Bonds']>0]   # select only those with at least one broken bond
        #print(len(bb_df))

        tot_bb = bb_df['Broken Bonds'].sum()
        input_tstep = float(getTimeStep("input.txt"))

        fnx = lambda row: row['x coord'] - xmax
        colx = bb_df.apply(fnx,axis=1)
        #bb_df['x coord shifted'] = bb_df['x coord'] - xmax
        bb_df = bb_df.assign(x_coord_shifted=colx.values)

        fny = lambda row: row['y coord'] - ymax
        coly = bb_df.apply(fny,axis=1)
        #bb_df['x coord shifted'] = bb_df['x coord'] - xmax
        bb_df = bb_df.assign(y_coord_shifted=coly.values)

        # get some info about the file
        input_viscosity = float(getViscosity("input.txt"))
        file_num = float(filename.split("experiment")[1].split(".")[0])  # first take the part after "experiment", then the one before the "."

        # name of the line in the plot
        labelName = "t/$\mu$=" + str('{:.1e}'.format(input_tstep*file_num/input_viscosity))

        first_value = 1/tot_bb*(sum(bb_df.x_coord_shifted**2))
        
        cov_matrix[0] = first_value
        second_value = 1/tot_bb*(sum(bb_df.x_coord_shifted*bb_df.y_coord_shifted))
        cov_matrix[1] = second_value
        cov_matrix[2] = second_value

        fourth_value = 1/tot_bb*(sum(bb_df.y_coord_shifted**2))
        cov_matrix[3] = fourth_value
        cov_matrix = np.reshape(cov_matrix, (-1, 2))  # from 1d to 2d (-1 means compute the proper shape automatically based on 2 columns)
        print(cov_matrix)

        w, v = np.linalg.eig(cov_matrix)  # eigenvalues and eigenvectors
        print("eigenvalues: " + str(w)+ ", \neigenvectors: "+ str(v))
        print(v[0][0])
        
        
        # TO DO: weight sum by n of bb

        if plot_figure:
            plt.figure()
            ax = plt.axes()
            sns.scatterplot(data=bb_df,x="x_coord_shifted",y="y_coord_shifted",hue="Broken Bonds",linewidth=0,alpha=0.8,marker="h",size=0.6,axes=ax).set_aspect('equal')
            ax.arrow(0,0,v[0][0],v[0][1], head_width=0.05, head_length=0.05)  # first eigenvector: arrow from 0,0 (origin) to the coordinates of 1st eigv
            ax.arrow(0,0,v[1][0],v[1][1], head_width=0.05, head_length=0.05)
            
        # x                  shift array by the max value so that the maximum is at zero and scale it 
        #data1_x = {'x': (x_array - ymax)*scale_factor,'Fluid Pressure': pressure_array_x, 'scale': dir_label,'time': labelName}  # save temporarily
        #data1_y = {'y': (y_array - ymax)*scale_factor,'Stress (MPa)': stress_array_y/1e6, 'scale': dir_label,'time': labelName}  # save temporarily
        #data1_y = {'y': (y_array - ymax)*scale_factor,'Fluid Pressure (MPa)': pressure_array_y/1e6, 'scale': dir_label,'time': labelName}  # save temporarily
        #df1_y = pd.DataFrame(data1_y)
        #df_y = pd.concat([df_y,df1_y], ignore_index=True) # append to old one

        # # y                  shift array by the max value so that the maximum is at zero
        # #data1_y = {'y': (y_array - xmax)*scale_factor,'Fluid Pressure': pressure_array_y, 'scale': dir_label,'time': labelName}  # save temporarily
        # data1_x = {'x': (x_array - xmax)*scale_factor,'Fluid Pressure (MPa)': pressure_array_x/1e6, 'scale': dir_label,'time': labelName}  # save temporarily
        # df1_x = pd.DataFrame(data1_x)
        # df_x = pd.concat([df_x,df1_x], ignore_index=True) # append to old one

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
