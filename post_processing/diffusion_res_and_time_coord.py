""" 
plot a profile
 1st plot: horizontal, 2nd plot: vertical

line:
  colour: time
  style: resolution - label is the size of the domain

takes the name of the subdirectories in a list - dir_labels
  
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

and the second one for y_coord

same as  diffusion_res_and_time, but uses coordinates instead of proximity threshold to get the points to plot
finds the max P, then uses the id of that point to get the coordinates of the points in the horizontal and vertical line

It's called "diffusion" because it was made to plot pressure diffusion in time
"""

import pandas as pd
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

var_to_plot = "Actual Movement"
# options: Pressure, Mean Stress, Actual Movement, Gravity, Porosity, Sigma_1, Sigma_2, Youngs Modulus
#         F_P_x, F_P_y, pf_grad_x, pf_grad_y, Original Movement, Movement in Gravity

# dir_labels = ['400','200']  # res400.elle, res200.elle
#dir_labels = ['00200', '00400','00600','00800','01000']
#dir_labels = ['02000','04000','06000','08000','10000'] 
dir_labels = ['00200', '00400','00600','00800','01000','02000','04000']#,'06000','08000','10000'] 
#dir_labels = ['01','03','05','07','09','11','13','15','17','19']
# dir_labels = ['01','02','03','04','05','06','07','08','09'] 

resolution = 200

dir_list = []; sigmas_top_true = []; sigmas_bot_true = []; sigmas_top_theor = []; sigmas_bot_theor = []
sigmas_top_ratio = []; sigmas_bot_ratio = []; sigmas_diff_theor = []; sigmas_diff_true = []; sigmas_diff_ratio = []
sizes = []

####   WARNING: (horizontal) offset is set to 50 instead of the central point (1/4 of domain)


for i in dir_labels:
    # dir_list.append('/nobackup/scgf/myExperiments/wavedec2022/wd_viscTest/vis_'+str(i))  
    dir_list.append('/nobackup/scgf/myExperiments/gaussJan2022/gj136/size'+str(i)) 
    # dir_list.append('/nobackup/scgf/myExperiments/threeAreas/through/th04/vis1e2_mR_'+str(i))  
    # dir_list.append('/nobackup/scgf/myExperiments/threeAreas/through/th38/vis1e2_mR_'+str(i))  
    # dir_list.append('/nobackup/scgf/myExperiments/gaussScaleFixFrac2/press_adjustGrav/press020_res200/press'+str(i))
    
print(dir_list)

df_x = pd.DataFrame()
df_y = pd.DataFrame()


# check if labels only contain numbers (it means they are the dimensions of the domain = scales)
if dir_labels[-1].startswith('size'):
    max_dir_size = float(dir_labels[-1])  # maximum size for the scaling of the plots: take the last directory (largest)
else:
    max_dir_size = 1
for dirnum,dir in enumerate(dir_list):
    """ loop through directories"""
    os.chdir(dir)
    if len(sorted(glob.glob("my_experiment*"))) < 4:  # if there aren't enough files, skip
        #if not os.path.exists(os.getcwd()+'/'+first_file):  # if it can't find the file after my_experiment00000.csv
        print('skipping '+os.getcwd())
        continue
    else:
        first_file = sorted(glob.glob("my_experiment*"))[3]
    

    myExp = pd.read_csv(first_file, header=0)

    if "size" in dir.split("/")[-1]:      # try to get the size automatically. If the last subdirectory contains the scale
        dir_label = dir_labels[dirnum]   # get the label that corresponds to the directory
        scale_factor = float(dir_label)/max_dir_size # factor for scaling the axes. Normalised by the maximum size (e.g. 1000)

        ## manually get the coordinates of the central point:
        meltYmin = float(getParameterFromLatte("input.txt","meltYmin"))
        # ymax = meltYmin/(resolution/2)   # in the form of 0.875   WHAT IT SHOULD BE
        ymax = meltYmin/(200/2)   # in the form of 0.875  WHAT WORKS WITH THE FIRST ONES (I haven't adapeted it yet)
        xmax = 0.3    # 0.5 if point is in the middle
        matches_x = np.where(np.isclose(myExp["x coord"],xmax,atol=3e-3)==True)[0] # vertical
        # print(matches_x) 
        # print(len(matches_x))
        matches_y = np.where(np.isclose(myExp["y coord"],ymax,atol=2e-3)==True)[0]
        # print(matches_y)
        # print(len(matches_y))
        # if dir_label == '08000':
        #     max_id = 45858   #  index of a point w coord 0.2975,0.980603
        #     xmax = myExp.loc[45707, 'x coord']
        #     ymax = myExp.loc[max_id, 'y coord']
        # if dir_label == '10000':
        #     max_id = 45858   #  index of a point w coord 0.2975,0.980603
        #     # max_id = 45707   #  index of the point with the maximum pressure value
        #     xmax = myExp.loc[max_id, 'x coord']
        #     ymax = myExp.loc[max_id, 'y coord']
            # offset = 100   # so that they are all centered around 0 on the x axis. It's the shift in the x direction.
        # max_id = np.where(np.isclose(myExp["x coord"],xmax,atol=1e-2) & np.isclose(myExp["y coord"],ymax,atol=1e-3)).index   #  index of the point with the maximum pressure value
        max_ids = myExp[(np.isclose(myExp["x coord"],xmax,atol=4e-3) & np.isclose(myExp["y coord"],ymax,atol=3e-3))].index   #  index of the point with the maximum pressure value
        # print(f'all matches for max_ids: {max_ids} ')
        if len(max_ids)>0:
            max_id = max_ids[0]
        else:
            max_id = max_ids

        # max_id = myExp_upper['Pressure'].idxmax()   #  index of the point with the maximum pressure value
        
        # what it should be:
        # offset = max_id%resolution   # so that they are all centered around 0 on the x axis. It's the shift in the x direction.
        # what works bc I haven't changed the location of the point yet:
        offset = max_id%200   # so that they are all centered around 0 on the x axis. It's the shift in the x direction.

    elif "press0" in dir.split("/")[-2]:   # If the second to last subdirectory contains the scale
        scale_factor = float(dir_label)/1000.0 # factor for scaling the axes. Normalised by the standard size
        dir_label = dir_labels[dirnum]   # get the label that corresponds to the directory
    # elif "vis" in dir.split("/")[-1]: 
    #     dir_label = str(int(dir_labels[dirnum])-4)   # get the label that corresponds to the directory. The viscosity is shifted by 4.
    #     scale_factor = 1 # default factor for scaling the axes. 
    else:  # set manually
        dir_label = dir_labels[dirnum]   # get the label that corresponds to the directory
        scale_factor = 1 # default factor for scaling the axes. 

    # open the first file after melt addition to find the max P


    # else:
        # xmax = myExp_upper.loc[myExp_upper['Pressure'].idxmax(), 'x coord']
        # ymax = myExp_upper.loc[myExp_upper['Pressure'].idxmax(), 'y coord']
        #max_id = myExp_upper['Pressure'].idxmax()   #  index of the point with the maximum pressure value
        # offset = max_id%resolution   # so that they are all centered around 0 on the x axis. It's the shift in the x direction.
        # print(f'offset: {offset}')
    # print(f'xmax: {xmax}, ymax: {ymax}')
        # offset = 50   # set manually
    
    # find the x coordinates that correspond to the max pressure, based on their index
    x_array = myExp.iloc[(max_id-offset):(max_id-offset)+resolution:1,myExp.columns.get_loc('x coord')] 
    y_array = myExp.iloc[offset::resolution,myExp.columns.get_loc('y coord')] # first: the point with coorinate = offset. Then every point above it (with a period of 'resolution') 

    for i,filename in enumerate(sorted(glob.glob("my_experiment*"))[0:6:5]): #[beg:end:step]  set which timesteps (based on FILE NUMBER) to plot
        myfile = Path(os.getcwd()+'/'+filename)  # build file name including path
        if myfile.is_file():
            myExp = pd.read_csv(filename, header=0)
            # get the values of the selected variable along a horizontal and a vertical line
            var_array_x =  myExp.iloc[(max_id-offset):(max_id-offset)+resolution:1,myExp.columns.get_loc(var_to_plot)] 
            var_array_y = myExp.iloc[offset::resolution,myExp.columns.get_loc(var_to_plot)]
            # FPx_array_x =  myExp.iloc[(max_id-offset):(max_id-offset)+resolution:1,myExp.columns.get_loc('F_P_x')] 
            # FPy_array_y = myExp.iloc[offset::resolution,myExp.columns.get_loc('F_P_y')]
            # pf_grad_x =  myExp.iloc[(max_id-offset):(max_id-offset)+resolution:1,myExp.columns.get_loc('pf_grad_x')] 
            # pf_grad_y = myExp.iloc[offset::resolution,myExp.columns.get_loc('pf_grad_y')]

            input_tstep = float(getTimeStep("input.txt"))
            input_viscosity = float(getViscosity("input.txt"))
            file_num = float(filename.split("experiment")[1].split(".")[0])  # first take the part after "experiment", then the one before the "."
            # name of the line
            #labelName = "t/$\mu$=" + str('{:.1e}'.format(input_tstep*file_num/input_viscosity))
            labelName = "t=" + str('{:.1e}'.format(input_tstep*file_num))

            # y                  shift array by the max value so that the maximum is at zero and scale it 
            data1_y = {'y': (y_array - ymax)*scale_factor,var_to_plot: var_array_y, 'scale': dir_label,'time': labelName}  # save temporarily. dictionary with y coord, the variable, the scale and time
            df1_y = pd.DataFrame(data1_y)
            df_y = pd.concat([df_y,df1_y], ignore_index=True) # append to old one

            # x                  shift array by the max value so that the maximum is at zero
            data1_x = {'x': (x_array - xmax)*scale_factor,var_to_plot: var_array_x, 'scale': dir_label,'time': labelName}  # save temporarily
            df1_x = pd.DataFrame(data1_x)
            df_x = pd.concat([df_x,df1_x], ignore_index=True) # append to old one
        # end of file loop
        # values now are referring to the last timestep that was loaded
    if var_to_plot == "Sigma_1":  # print out some values about sigma_1
        dom_size = float(getParameterFromLatte('input.txt','Scale'))
        depth = getDepth("input.txt")
        sigma_1_top_theor = 2968.94*9.8*float(depth)  # rho * g * depth
        sigma_1_bot_theor = 2968.94*9.8*(float(depth)+float(dom_size))  # rho * g * (depth+size)
        print(f'scale: {dom_size}')
        print(f'bottom theor: {sigma_1_bot_theor/1e6:.2f} MPa, bottom true: {-var_array_y.values[1]/1e6}. true/theor = {-var_array_y.values[1]/sigma_1_bot_theor:.4f}')
        print(f'top theor:   {sigma_1_top_theor/1e6:.2f} MPa, top true:    {-var_array_y.values[-2]/1e6}. true/theor = {-var_array_y.values[-2]/sigma_1_top_theor:.4f}')
        #print(f'bottom: {-var_array_y.values[0]/1e6}. true/theor = {-var_array_y.values[0]/sigma_1_bot_theor}')
        #print(f'top:    {-var_array_y.values[-1]/1e6}. true/theor = {-var_array_y.values[-1]/sigma_1_top_theor}')
        print(f'bot-top:    {-(var_array_y.values[1]-var_array_y.values[-2])/1e6}. theor = {(sigma_1_bot_theor-sigma_1_top_theor)/1e6:.4f}\n')
        sigmas_top_true.append(-var_array_y.values[-2])
        sigmas_bot_true.append(-var_array_y.values[1])
        sigmas_top_theor.append(sigma_1_top_theor)
        sigmas_bot_theor.append(sigma_1_bot_theor)
        sigmas_top_ratio.append(-var_array_y.values[-2]/sigma_1_top_theor)
        sigmas_bot_ratio.append(-var_array_y.values[1]/sigma_1_bot_theor)
        sigmas_diff_true.append(var_array_y.values[-2]-var_array_y.values[1])   # top - bottom (they're <0, so difference is >0)
        sigmas_diff_theor.append(sigma_1_bot_theor-sigma_1_top_theor)   # bottom - top (they're >0)
        sigmas_diff_ratio.append((var_array_y.values[-2]-var_array_y.values[1])/(sigma_1_bot_theor-sigma_1_top_theor))   # bottom - top (they're >0)
        sizes.append(float(dom_size))
    # end of dir loop

# g_x = sns.lineplot(data=df_x ,x="x",y='Mean Stress (MPa)',ax=ax1,hue='time',style='scale',alpha=0.5)  # or sns.scatterplot
# g_y = sns.lineplot(data=df_y ,x="y",y='Mean Stress (MPa)',ax=ax2,hue='time',style='scale',alpha=0.5)
# ax3 = g_y.twinx()
# g_y_1 = sns.lineplot(data=df_y ,x="y",y='F_P_y',ax=ax3,hue='time',markers=True,alpha=0.5)
# g_y_1 = sns.scatterplot(data=df_y ,x="y",y='Porosity',ax=ax3,hue='time',alpha=0.5)

os.chdir('..')

def set_axes_options(fig):
    """ set some styling that is in common between all plots of type true-theor:
    ticks, legend and title """
    for ax in fig.axes:
        ax.set_xticks(ticks=[float(x) for x in sizes])  # define list of ticks
        ax.xaxis.set_tick_params(labelsize=10, rotation=90)
        ax.legend()
    fig.suptitle(os.getcwd())

if True:
    fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2)
    g_x = sns.lineplot(data=df_x ,x="x",y=var_to_plot,ax=ax1,hue='time',style='scale',alpha=0.5)
    g_y = sns.lineplot(data=df_y ,x="y",y=var_to_plot,ax=ax2,hue='time',style='scale',alpha=0.5)
    ax1.set_title("Horizontal Profile")
    #ax1.set_xlim([-0.51,+0.51])  # zoom in. Limits are location of max pressure +- 0.05
    #ax2.set_xlim([-1.01,+0.12])  # zoom in. Limits are location of max pressure +- 0.05
    # ax2.set_ylim([2.942975e7,2.943020e7])  # for pressure
    ax2.set_title("Vertical Profile")

    g_y.legend_.remove()
    

    ax1.legend(fancybox=True, ncol=1)   # legend for the horizontal line plot
    # save:
    #plt.savefig("gaussScale50-100-200_diff_hrz-vrt.png", dpi=600,transparent=True)
    plt.tight_layout()
    fig.suptitle(os.getcwd()) 
    plt.show()
if var_to_plot == "Sigma_1":
    if True:
        fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2)
        ax1.plot(sizes,sigmas_top_true,'-o',color='blue',label='top true')
        ax1.plot(sizes,sigmas_bot_true,'-o',color='green',label='bottom true')
        ax2.plot(sizes,sigmas_diff_true,'-o',label='difference true')
        ax1.plot(sizes,sigmas_top_theor,'--',color='blue',label='top theoretical')
        ax1.plot(sizes,sigmas_bot_theor,'--',color='green',label='bottom theoretical')
        ax2.plot(sizes,sigmas_diff_theor,'--',label='difference theoretical')
        set_axes_options(fig)
        plt.show()

    if True:  # plot ratios
        fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2)
        ax1.plot(sizes,sigmas_top_ratio,'-o',color='blue',label='top ratio')
        ax1.plot(sizes,sigmas_bot_ratio,'-o',color='green',label='bottom ratio')
        ax2.plot(sizes,sigmas_diff_ratio,'-o',label='diff ratio')
        for ax in fig.axes:
            ax.axhline(y=1.0, linestyle='--',color='orange')
        set_axes_options(fig)
        # ax1.set_xticks(ticks=[float(x) for x in dir_labels])  # define list of ticks
        # ax2.set_xticks(ticks=[float(x) for x in dir_labels])  # define list of ticks
        # ax1.xaxis.set_tick_params(labelsize=10, rotation=90)
        # ax2.xaxis.set_tick_params(labelsize=10, rotation=90)
        # ax1.legend()
        # ax2.legend()
        plt.show()


