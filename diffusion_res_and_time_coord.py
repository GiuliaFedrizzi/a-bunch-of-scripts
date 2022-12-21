""" plot a line = pressure diffusion
 1st plot: horizontal, 2nd plot: vertical

line:
  colour: time
  style: resolution

can plot from different directories (e.g. sigma, resolution)
builds 2 pandas dataframes of the type:

      x_coord    pressure  dir      time
0     0.00125  40024800.0    1  t=1.0e+06
1     0.00375  40024800.0    1  t=1.0e+06
2     0.00625  40024800.0    1  t=1.0e+06
3     0.00875  40024800.0    1  t=1.0e+06
4     0.01125  40024800.0    1  t=1.0e+06
...       ...         ...  ...        ...
1195  0.98875  40024800.0    1  t=9.0e+06
1196  0.99125  40024800.0    1  t=9.0e+06
1197  0.99375  40024800.0    1  t=9.0e+06
1198  0.99625  40024800.0    1  t=9.0e+06
1199  0.99875  40024800.0    1  t=9.0e+06

and the second one for x_coord

same as  diffusion_res_and_time, but uses coordinates instead of proximity threshold to get the points to plot
finds the max P, then uses the id of that point to get the coordinates of the points in the horizontal and vertical line

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

# import functions from external file
from useful_functions import * 

# dir_list = ['/nobackup/scgf/myExperiments/gaussTime/sigma_3_0/sigma_3_0_gaussTime05/tstep04_5_1e5',
#      '/nobackup/scgf/myExperiments/gaussScale/scale200_r200/sigma_3_0/sigma_3_0_gaussTime05/tstep04_5_1e5']

# sigma_str = "/sigma_2_0"
# time_str = '$10^{-2}$'


# dir_labels = ['400','200']  # res400.elle, res200.elle
dir_labels = ['020','030','040','050','060','070','080','090','100']
# dir_labels = ['100']
#dir_labels = ['01','03','05','07','09','11','13','15','17','19']
#dir_labels = ['11']#,'03','05','07','09','11','13','15','17','19']

resolution = 200

dir_list = []

for i in dir_labels:
    dir_list.append('/nobackup/scgf/myExperiments/gaussScaleFixFrac2/g2_13_rad_wGrav200/size'+str(i))
    # dir_list.append('/nobackup/scgf/myExperiments/gaussScaleFixFrac2/press_adjustGrav/press020_res200/press'+str(i))
    # dir_list.append('/nobackup/scgf/myExperiments/gaussScaleFixFrac2/g2_02_trueGrad400/size'+str(i))
    


fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2)
first_file = 'my_experiment00100.csv'
df_x = pd.DataFrame()
df_y = pd.DataFrame()

dirnum = 0
for dir in dir_list:
    print(dir)
    """ loop through directories"""
    os.chdir(dir)
    dirnum+=1 # counter for directories
    dir_label = dir_labels[dirnum-1]   # get the label that corresponds to the directory
    
    if "size" in dir.split("/")[-1]:      # try to get the size automatically. If the last subdirectory contains the scale
        scale_factor = float(dir_label)/100.0 # factor for scaling the axes. Normalised by the standard size = 100
    elif "press0" in dir.split("/")[-2]:   # If the second to last subdirectory contains the scale
        scale_factor = float(dir_label)/100.0 # factor for scaling the axes. Normalised by the standard size = 100

    else:  # set manually
        scale_factor = 20.0/100.0 # factor for scaling the axes. Normalised by the standard size = 100

    print("scale: ",str(scale_factor))
    # open the first file after melt addition to find the max P
    myExp = pd.read_csv(first_file, header=0)
    xmax = myExp.loc[myExp['Pressure'].idxmax(), 'x coord']
    ymax = myExp.loc[myExp['Pressure'].idxmax(), 'y coord']
    max_id = myExp['Pressure'].idxmax()   #  index of the point with the maximum pressure value
    
    offset = max_id%resolution   # so that they are all centered around 0 on the x axis

    # find the x coordinates that correspond to the max pressure, based on their index
    x_array = myExp.iloc[(max_id-offset):(max_id-offset)+resolution:1,myExp.columns.get_loc('x coord')] 
    y_array = myExp.iloc[offset::resolution,myExp.columns.get_loc('y coord')] # first: the point with coorinate = offset. Then every point above it (with a period of 'resolution') 

    for i,filename in enumerate(sorted(glob.glob("my_experiment*"))[10:70:20]): #[beg:end:step]  set which timesteps (based on FILE NUMBER) to plot
    # for i,filename in enumerate(sorted(glob.glob("my_experiment*"))[0:16:1]): #[beg:end:step]  set which timesteps (based on FILE NUMBER) to plot
        myfile = Path(os.getcwd()+'/'+filename)  # build file name including path
        if myfile.is_file():
            myExp = pd.read_csv(filename, header=0)
            # pressure_array_x =  myExp.iloc[(max_id-offset):(max_id-offset)+resolution:1,myExp.columns.get_loc('Pressure')] 
            # pressure_array_y = myExp.iloc[offset::resolution,myExp.columns.get_loc('Pressure')]
            # poro_array_x =  myExp.iloc[(max_id-offset):(max_id-offset)+resolution:1,myExp.columns.get_loc('Porosity')] 
            # poro_array_y = myExp.iloc[offset::resolution,myExp.columns.get_loc('Porosity')]
            # stress_array_x =  myExp.iloc[(max_id-offset):(max_id-offset)+resolution:1,myExp.columns.get_loc('Mean Stress')] 
            # stress_array_y = myExp.iloc[offset::resolution,myExp.columns.get_loc('Mean Stress')]
            # sigma1_array_x =  myExp.iloc[(max_id-offset):(max_id-offset)+resolution:1,myExp.columns.get_loc('Sigma_1')] 
            # sigma1_array_y = myExp.iloc[offset::resolution,myExp.columns.get_loc('Sigma_1')]
            # sigma2_array_x =  myExp.iloc[(max_id-offset):(max_id-offset)+resolution:1,myExp.columns.get_loc('Sigma_2')] 
            # sigma2_array_y = myExp.iloc[offset::resolution,myExp.columns.get_loc('Sigma_2')]
            # FPx_array_x =  myExp.iloc[(max_id-offset):(max_id-offset)+resolution:1,myExp.columns.get_loc('F_P_x')] 
            # FPy_array_y = myExp.iloc[offset::resolution,myExp.columns.get_loc('F_P_y')]
            # real_radius =  myExp.iloc[(max_id-offset):(max_id-offset)+resolution:1,myExp.columns.get_loc('real_radius')] 
            # pf_grad_x =  myExp.iloc[(max_id-offset):(max_id-offset)+resolution:1,myExp.columns.get_loc('pf_grad_x')] 
            # pf_grad_y = myExp.iloc[offset::resolution,myExp.columns.get_loc('pf_grad_y')]
            # box_x =  myExp.iloc[(max_id-offset):(max_id-offset)+resolution:1,myExp.columns.get_loc('fluid_box')] 
            # box_y = myExp.iloc[offset::resolution,myExp.columns.get_loc('fluid_box')]
            # real_radius_x =  myExp.iloc[(max_id-offset):(max_id-offset)+resolution:1,myExp.columns.get_loc('real_radius')] 
            grav_y = myExp.iloc[offset::resolution,myExp.columns.get_loc('Gravity')]
            grav_x =  myExp.iloc[(max_id-offset):(max_id-offset)+resolution:1,myExp.columns.get_loc('Gravity')] 
            # print(dir_label,str(pf_grad_x[pf_grad_x>0].mean()),str(pf_grad_y[pf_grad_y>0].mean()))
            
            input_tstep = float(getTimeStep("input.txt"))
            input_viscosity = float(getViscosity("input.txt"))
            file_num = float(filename.split("experiment")[1].split(".")[0])  # first take the part after "experiment", then the one before the "."
            # name of the line
            #labelName = "t/$\mu$=" + str('{:.1e}'.format(input_tstep*file_num/input_viscosity))
            labelName = "t=" + str('{:.1e}'.format(input_tstep*file_num))

            # y                  shift array by the max value so that the maximum is at zero and scale it 
            # data1_y = {'y': (y_array - ymax)*scale_factor,'F_P_y': FPy_array_y, 'scale': dir_label,'time': labelName}  # save temporarily
            # data1_y = {'y': (y_array - ymax)*scale_factor,'Sigma_1 (MPa)': sigma1_array_y/1e6, 'scale': dir_label,'time': labelName}  # save temporarily
            # data1_y = {'y': (y_array - ymax)*scale_factor,'P Gradient': pf_grad_y, 'scale': dir_label,'time': labelName}  # save temporarily
            # data1_y = {'y': (y_array - ymax)*scale_factor,'Mean Stress (MPa)': stress_array_y/1e6, 'scale': dir_label,'time': labelName}  # save temporarily
            # data1_y = {'y': (y_array - ymax)*scale_factor,'Fluid Pressure (MPa)': pressure_array_y/1e6, 'scale': dir_label,'time': labelName}  # save temporarily
            # data1_y = {'y': (y_array - ymax)*scale_factor,'box': box_y, 'Porosity': poro_array_y, 'scale': dir_label,'time': labelName}  # save temporarily
            # data1_y = {'y': (y_array - ymax)*scale_factor,'Porosity': poro_array_y, 'scale': dir_label,'time': labelName}  # save temporarily
            data1_y = {'y': (y_array - ymax)*scale_factor,'Gravity': grav_y, 'scale': dir_label,'time': labelName}  # save temporarily
            df1_y = pd.DataFrame(data1_y)
            df_y = pd.concat([df_y,df1_y], ignore_index=True) # append to old one

            # x                  shift array by the max value so that the maximum is at zero
            # data1_x = {'x': (x_array - xmax)*scale_factor,'F_P_x': FPx_array_x, 'scale': dir_label,'time': labelName}  # save temporarily
            # data1_x = {'x': (x_array - xmax)*scale_factor,'P Gradient': pf_grad_x, 'scale': dir_label,'time': labelName}  # save temporarily
            # data1_x = {'x': (x_array - xmax)*scale_factor,'Sigma_1': sigma1_array_x, 'scale': dir_label,'time': labelName}  # save temporarily
            # data1_x = {'x': (x_array - xmax)*scale_factor,'Sigma_1 (MPa)': sigma1_array_x/1e6, 'scale': dir_label,'time': labelName}  # save temporarily
            # data1_x = {'x': (x_array - xmax)*scale_factor,'Mean Stress (MPa)': stress_array_x/1e6, 'scale': dir_label,'time': labelName}  # save temporarily
            # data1_x = {'x': (x_array - xmax)*scale_factor,'Fluid Pressure (MPa)': pressure_array_x/1e6, 'scale': dir_label,'time': labelName}  # save temporarily
            # data1_x = {'x': (x_array - xmax)*scale_factor,'box': box_x, 'scale': dir_label,'time': labelName}  # save temporarily
            # data1_x = {'x': (x_array - xmax)*scale_factor,'Porosity': poro_array_x, 'scale': dir_label,'time': labelName}  # save temporarily
            data1_x = {'x': (x_array - xmax)*scale_factor,'Gravity': grav_x, 'scale': dir_label,'time': labelName}  # save temporarily
            df1_x = pd.DataFrame(data1_x)
            df_x = pd.concat([df_x,df1_x], ignore_index=True) # append to old one

g_x = sns.lineplot(data=df_x ,x="x",y='Gravity',ax=ax1,hue='time',style='scale',alpha=0.5)  # or sns.scatterplot
g_y = sns.lineplot(data=df_y ,x="y",y='Gravity',ax=ax2,hue='time',style='scale',alpha=0.5)
# ax3 = g_y.twinx()
# g_y_1 = sns.lineplot(data=df_y ,x="y",y='F_P_y',ax=ax3,hue='time',markers=True,alpha=0.5)
# g_y_1 = sns.scatterplot(data=df_y ,x="y",y='Porosity',ax=ax3,hue='time',alpha=0.5)

#g_x = sns.lineplot(data=df_x ,x="x",y="Fluid Pressure (MPa)",ax=ax1,hue='time',style='scale',alpha=0.5)
#g_y = sns.lineplot(data=df_y ,x="y",y="Fluid Pressure (MPa)",ax=ax2,hue='time',style='scale',alpha=0.5)
ax1.set_title("Horizontal Profile")
ax1.set_xlim([-0.12,+0.12])  # zoom in. Limits are (location of max pressure) +- 0.1
ax2.set_xlim([-0.12,+0.12])  # zoom in. Limits are location of max pressure +- 0.1
# ax1.set_xlim([-0.08,+0.08])  # zoom in. Limits are (location of max pressure) +- 0.1
# ax2.set_xlim([-0.08,0.08])  # zoom in. Limits are location of max pressure +- 0.1
# ax2.set_ylim([60000,80000])  # zoom in.
ax2.set_title("Vertical Profile")

# g_x.legend_.remove()
g_y.legend_.remove()
os.chdir('..')
fig.suptitle(os.getcwd()) # get the part of the path that is after "myExperiments/"
#fig.suptitle("$\sigma$ = "+sigma_str.replace("/sigma_","").replace("_",".")+", tstep = " + time_str.split("_")[1]) # get the part of the path that is after "myExperiments/"
#fig.suptitle("Fluid Pressure") # get the part of the path that is after "myExperiments/"

# LEGEND:
# get the position of the plot so I can add the legend to it 
# box = ax2.get_position()

# upper left = the point that I am setting wiht the second argument
# ax2.legend(loc='center left',bbox_to_anchor=(1.2,0.5),fancybox=True, ncol=1)   # legend for the vertical line plot
# ax2.legend(fancybox=True, ncol=1)   # legend for the vertical line plot
ax1.legend(fancybox=True, ncol=1)   # legend for the horizontal line plot
# save:
#os.chdir('/nobackup/scgf/myExperiments/gaussScale')
#plt.savefig("gaussScale50-100-200_diff_hrz-vrt.png", dpi=600,transparent=True)
plt.tight_layout()
plt.show()

