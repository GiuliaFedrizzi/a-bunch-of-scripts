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
import re

# import functions from external file
from useful_functions import * 

var_to_plot = "Pressure"
# options: Pressure, Mean Stress, Actual Movement, Gravity, Porosity, Sigma_1, Sigma_2, Youngs Modulus, Differential Stress,Permeability
#         F_P_x, F_P_y, pf_grad_x, pf_grad_y, Original Movement, Movement in Gravity, Smooth function, area_par_fluid
#         gauss_scaling_par, gauss_scaling_par_sum, gauss_scaling_par_n_tot, xy_melt_point
#         Youngs in Gravity, Poisson in Gravity, Youngs Modulus E, Youngs Modulus Par, Youngs Modulus Real, fg, poisson_ratio
#         Broken Bonds


# dir_labels = ['scale400/young1','scale400/young2','scale800/young1','scale800/young2','scale1000/young2']
# dir_labels = ['scale400','scale800']#,'scale1000']
# dir_labels = ['res200/depth1000/rb0.006','res200/depth1000/rb0.01','res200/depth1000/rb0.02']#,'res400/depth1000/rb0.007','res400/depth1000/rb0.01']#,'res400/rb0.01']#,'res400/rb0.03']
# dir_labels = ['res400/depth1000/rb0.01']#'res400/depth1000/rb0.003','res400/depth1000/rb0.007','res400/depth1000/rb0.01']#,'res400/rb0.01']#,'res400/rb0.03']
# dir_labels = ['grad0.2','grad0.4','grad0.6']
# dir_labels = ['rb0.006','rb0.01','rb0.02']
# dir_labels = ['tw12/rt0.1/pincr1e6','tw12/rt0.5/pincr1e6'] # 'tw12/rt0.01/pincr1e6','tw12/rt0.03/pincr1e6',
# dir_labels = ['si08','si10']#,'rb1.0'] # 'rb0.01',  
# dir_labels = ['rt0.01','rt0.03']#,'rt0.05'] 
# dir_labels = ['op13/rt0.03/threads1','op18/rt0.03/threads2','op18/rt0.03/threads4']#,'threads8','threads16']#,'rt0.05'] 
# dir_labels = ['rt0.08/mrate0.0005']#,'threads8','threads16']#,'rt0.05'] 
dir_labels = ['vis1e15_mR_03']#,'threads8','threads16']#,'rt0.05'] 
# dir_labels = ['young0.5', 'young1']
# dir_labels = ['rb0.03', 'rb0.05','rb0.5']
# dir_labels = ['bs45/depth1000/young0.7/rb0.5','bs45/depth1500/young0.7/rb0.5','bs45/depth2000/young0.7/rb0.5']
# dir_labels = ['young0.5','young0.7','young1']

# my_labels = ['p52, 0.001','p54, 0.0005','p55, 0.0001']#,'p49, 0.005'] # leave empty for default labels (= dir labels)
# my_labels = ['rt1 = 0.5, rt2 = 0.5','rt1 = 0.05, rt2 = 0.5','rt1 = 0.1, rt2 = 0.5'] # leave empty for default labels (= dir labels)
my_labels = [] # leave empty for default labels (= dir labels)
# my_labels = ['grad0.3','grad0.4'] # leave empty for default labels (= dir labels)

# resolution = 200 # now extracted automatically

dir_list = []; sigmas_top_true = []; sigmas_bot_true = []; sigmas_top_theor = []; sigmas_bot_theor = []
sigmas_top_ratio = []; sigmas_bot_ratio = []; sigmas_diff_theor = []; sigmas_diff_true = []; sigmas_diff_ratio = []
sizes = []


######  first complete relaxation  ---> Saving file: #####     

for i in dir_labels:
    # dir_list.append('/nobackup/scgf/myExperiments/threeAreas/prod/prt/prt16/'+str(i)+'/visc_1_1e1/vis1e1_mR_01')  
    # dir_list.append('/nobackup/scgf/myExperiments/gravity_x/'+str(i))  
    # dir_list.append('/nobackup/scgf/myExperiments/gravity_x/gy11/'+str(i)+'/gx_02/')  
    # dir_list.append('/nobackup/scgf/myExperiments/threeAreas/prod/prt/background_stress/bs37/depth0500/'+i+'/rb0.5')
    # dir_list.append('/nobackup/scgf/myExperiments/threeAreas/prod/prt/background_stress/'+i)
    # dir_list.append('/nobackup/scgf/myExperiments/threeAreas/through/thr/thr01/rt0.01/'+i)
    # dir_list.append('/nobackup/scgf/myExperiments/threeAreas/through/thr/thr02/'+i+'/pincr1e2')
    # dir_list.append('/nobackup/scgf/myExperiments/threeAreas/prod/prt/background_stress/bs23/'+i+'/depth0100/young2/rb0.03')
    # dir_list.append('/nobackup/scgf/myExperiments/threeAreas/prod/prt/singleInjection/si22/'+i) #'rb0.003'
    dir_list.append('/nobackup/scgf/myExperiments/threeAreas/prod/prt/prt45/rt0.5/visc_1_1e15/'+i)
    # dir_list.append('/nobackup/scgf/myExperiments/threeAreas/through/thprod/tp12/'+i)
    # dir_list.append('/nobackup/scgf/myExperiments/threeAreas/through/thprod/single/ts04/'+i+'/mrate1e6')
    # dir_list.append('/nobackup/scgf/myExperiments/optimise/'+i)
    
print(dir_list)

f1=-1  # first file to plot. They account for "my_experiment-0003.csv" as the first file in dir :::  -1  =  0
f2= 0  # second file. if f2 = 5 -> my_experiment00500.csv
step=1

df_x = pd.DataFrame()
df_y = pd.DataFrame()

# Get the number from the filename
def extract_number(filename):
    match = re.search(r"(\d+)", filename)
    if match:
        return int(match.group(1))
    return 0

dir_counter_200=0
dir_counter_400=0
res200_flag=0
res400_flag=0

# check if labels only contain numbers (it means they are the dimensions of the domain = scales)
if dir_labels[-1].startswith('size'):
    max_dir_size = float(dir_labels[-1])  # maximum size for the scaling of the plots: take the last directory (largest)
else:
    max_dir_size = 1
for dirnum,dir in enumerate(dir_list):
    """ loop through directories"""
    os.chdir(dir)
    print(f'dir is {dir}')
    if len(sorted(glob.glob("my_experiment*"))) < f1:  # if there aren't enough files, skip
        #if not os.path.exists(os.getcwd()+'/'+first_file):  # if it can't find the file after my_experiment00000.csv
        print('skipping '+os.getcwd())
        continue
    else:
        resolution = int(getResolution())
        # relaxed_file = getWhenItRelaxed("latte.log")
        # print(f'First file after complete relaxation: {relaxed_file}')  # not working
        # --- for paths that contain "res200" and "res400": use colour to differentiate between resolutions, not time (default)
        current_path_components = os.getcwd().split(os.sep)
        if 'res200' in current_path_components:
            has_res_directory = True
            res_directory = 'res200'
            res200_flag = 1
            res400_flag = 0
            dir_counter_200+=1
            rb=current_path_components[-1]
        elif 'res400' in current_path_components:
            res200_flag = 0
            res400_flag = 1
            has_res_directory = True
            res_directory = 'res400'
            dir_counter_400+=1
            rb=current_path_components[-1]
        else:
            has_res_directory = False
            res_directory = 'no dir'
            rb=0
        first_file = sorted(glob.glob("my_experiment*"))[1]
        dom_size = float(getParameterFromLatte('input.txt','Scale'))
        # print(f'scale: {dom_size}')

        myExp = pd.read_csv(first_file, header=0)
        # with open("latte.log") as iFile:
        #     count = 1   # counter
        #     relax_thresh=[]  # initialise vector of relaxation thresholds (there should be 2)
        #     for num, line in enumerate(iFile,1):
        #         if "Changing relaxthresh" in line:
        #             rel = line.split(" ")[-1]
        #             rel = rel.replace("\n","")
        #             relax_thresh.append(rel)  # split around spaces, take the last (=relax thresh value)
        #             count+=1
        #             if count==2:  # stop after finding the second occurrence
        #                 continue  
        # print(f'relax thresholds: {relax_thresh[0]}, {relax_thresh[1]}')
        if "size" in dir.split("/")[-1]:      # try to get the size automatically. If the last subdirectory contains the scale
            dir_label = dir_labels[dirnum]   # get the label that corresponds to the directory
            scale_factor = float(dir_label)/max_dir_size # factor for scaling the axes. Normalised by the maximum size (e.g. 1000)
       
        elif "press0" in dir.split("/")[-2]:   # If the second to last subdirectory contains the scale
            scale_factor = float(dir_label)/1000.0 # factor for scaling the axes. Normalised by the standard size
            dir_label = dir_labels[dirnum]   # get the label that corresponds to the directory
        # elif "vis" in dir.split("/")[-1]: 
        #     dir_label = str(int(dir_labels[dirnum])-4)   # get the label that corresponds to the directory. The viscosity is shifted by 4.
        #     scale_factor = 1 # default factor for scaling the axes. 
        else:  # set manually
            dir_label = dir_labels[dirnum]   # get the label that corresponds to the directory
            scale_factor = dom_size # default factor for scaling the axes. 

        # get the maximum height of the domain (will be something like 0.99). Depends on how 'compressed' the domain has been
        domain_max_y = max(myExp["y coord"])
        domain_min_y = min(myExp["y coord"])
        domain_max_x = max(myExp["x coord"])
        domain_min_x = min(myExp["x coord"])
        ## manually get the coordinates of the central point:
        # meltYmin = float(getParameterFromLatte("input.txt","meltYmin"))
        # ymax = meltYmin/(resolution/2)   #  WHAT IT SHOULD BE
        # ymax = meltYmin/(200/2)   # in real units (meters)
        ymax = 0.505# * domain_max_y <-- no, don't scale because the melt point is chosen in the fluid lattice grid (fixed)
        # print(f'max y coord: {max(myExp["y coord"])}')
        print(f'ymax: {ymax}')
        xmax = 0.503    # 0.5 if point is in the middle (or 0.502)

        # define tolerances when looking for points along a vertical and a horizontal line starting from the chosen point
        #  1.15*resolution = 230 or 460, so the number of particles in the y direction. Using this, 
        tolerance_y = (domain_max_y-domain_min_y)/(1.15*resolution)
        tolerance_x = (domain_max_x-domain_min_x)/resolution
        # print(f'tolerance in y: {tolerance_y}')
        # print(f'tolerance in x: {tolerance_x}')
        matches_x = np.where(np.isclose(myExp["x coord"],xmax,atol=tolerance_x)==True)[0] # vertical
        print(f'x matches: {len(matches_x)}')
        matches_y = np.where(np.isclose(myExp["y coord"],ymax,atol=tolerance_y)==True)[0]
        # print(matches_y) 
        print(f'y matches: {len(matches_y)}')
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
        max_ids = myExp[(np.isclose(myExp["x coord"],xmax,atol=tolerance_x) & np.isclose(myExp["y coord"],ymax,atol=tolerance_y))].index   #  index of the point with the maximum pressure value
        # print(f'all matches for max_ids: {max_ids} ')
        if len(max_ids)>0:
            max_id = max_ids[0]
        else:
            max_id = max_ids
        # print(f'max_id {max_id}')
        # max_id = myExp_upper['Pressure'].idxmax()   #  index of the point with the maximum pressure value
        
        # what it should be:
        # offset = max_id%resolution   # so that they are all centered around 0 on the x axis. It's the shift in the x direction.
        # what works bc I haven't changed the location of the point yet:
        offset = max_id%resolution+1   # so that they are all centered around 0 on the x axis. It's the shift in the x direction.
        # print(f'offset {offset}')


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

        real_radius = getParameterFromInput('input.txt','Scale')/resolution  # real_radius is scale/res
        # print(f'real radius: {real_radius}')

        # for i,filename in enumerate(sorted(glob.glob("my_experiment*"))[f1+1:f2+2:(f2-f1)]): #[beg:end:step]  set which timesteps (based on FILE NUMBER) to plot. first and second file are defined at the beginning
        for i,filename in enumerate(sorted(glob.glob("my_experiment*"),key=extract_number)[f1+1:f2+2:step]): #[beg:end:step]  set which timesteps (based on FILE NUMBER) to plot. first and second file are defined at the beginning
            myfile = Path(os.getcwd()+'/'+filename)  # build file name including path
            if myfile.is_file():
                print(filename)
                myExp = pd.read_csv(filename, header=0)
                # get the values of the selected variable along a horizontal and a vertical line
                var_array_x =  myExp.iloc[(max_id-offset):(max_id-offset)+resolution:1,myExp.columns.get_loc(var_to_plot)] 
                var_array_y = myExp.iloc[offset::resolution,myExp.columns.get_loc(var_to_plot)]
                # if dirnum == 0:
                #     var_array_x = var_array_x*2
                #     var_array_y = var_array_y*2
                input_tstep = float(getTimeStep("input.txt"))
                # print(f'timestep: {input_tstep}')
                input_viscosity = float(getViscosity("input.txt"))
                file_num = float(filename.split("experiment")[1].split(".")[0])  # first take the part after "experiment", then the one before the "."
                # name of the line
                #labelName = "t/$\mu$=" + str('{:.1e}'.format(input_tstep*file_num/input_viscosity))
                labelName = "t=" + str('{:.1e}'.format(input_tstep*file_num))

                # y                  shift array by the max value so that the maximum is at zero and scale it 
                data1_y = {'y': (y_array - ymax)*scale_factor,var_to_plot: var_array_y, 'scale': dir_label,'time': labelName,'res':res_directory,'dir_counter':dir_counter_200*res200_flag+dir_counter_400*res400_flag,'rel_thr':rb}  # save temporarily. dictionary with y coord, the variable, the scale and time
                df1_y = pd.DataFrame(data1_y)
                df_y = pd.concat([df_y,df1_y], ignore_index=True) # append to old one

                # x                  shift array by the max value so that the maximum is at zero
                data1_x = {'x': (x_array - xmax)*scale_factor,var_to_plot: var_array_x, 'scale': dir_label,'time': labelName,'res':res_directory,'dir_counter':dir_counter_200*res200_flag+dir_counter_400*res400_flag,'rel_thr':rb}  # save temporarily
                df1_x = pd.DataFrame(data1_x)
                df_x = pd.concat([df_x,df1_x], ignore_index=True) # append to old one
            # end of file loop
            # values now are referring to the last timestep that was loaded
        # calculate and print out theoretical and true values of stresses
        if var_to_plot ==  "Sigma_1" or var_to_plot ==  "Sigma_2":
            depth = getDepth("input.txt")
            # solid_density = getDensity()
            solid_density = 3000
            # print(f'solid_density {solid_density}')
            sigma_top_theor = 0.66666 * solid_density*9.8*(float(depth)+2*float(real_radius))  # rho * g * depth+first row  (height of first row = 2*real_radius)
            sigma_bot_theor = 0.66666 * solid_density*9.8*(float(depth)+float(dom_size))  # rho * g * (depth+size)

            sizes.append(float(dom_size))
            if var_to_plot == "Sigma_1":  # print out some values about sigma_1
                print("Sigma 1")

            elif var_to_plot == "Sigma_2":
                print("Sigma 2")
                sigma_top_theor /=2   # lateral stress is half the vertical stress in axia strain
                sigma_bot_theor /=2

            print(f'bottom theor: {sigma_bot_theor/1e6:.2f} MPa, bottom true: {-var_array_y.values[1]/1e6}. true/theor = {-var_array_y.values[1]/sigma_bot_theor:.4f}')
            print(f'top theor:   {sigma_top_theor/1e6:.2f} MPa, top true:    {-var_array_y.values[-2]/1e6}. true/theor = {-var_array_y.values[-2]/sigma_top_theor:.4f}')
            #print(f'bottom: {-var_array_y.values[0]/1e6}. true/theor = {-var_array_y.values[0]/sigma_bot_theor}')
            #print(f'top:    {-var_array_y.values[-1]/1e6}. true/theor = {-var_array_y.values[-1]/sigma_top_theor}')
            print(f'bot-top: theor = {(sigma_bot_theor-sigma_top_theor)/1e6:.4f}, true {-(var_array_y.values[1]-var_array_y.values[-2])/1e6}, ratio: {(-(var_array_y.values[1]-var_array_y.values[-2]))/(sigma_bot_theor-sigma_top_theor)}\n')
            sigmas_top_true.append(-var_array_y.values[-2]); sigmas_bot_true.append(-var_array_y.values[1])
            sigmas_top_theor.append(sigma_top_theor); sigmas_bot_theor.append(sigma_bot_theor)
            sigmas_top_ratio.append(-var_array_y.values[-2]/sigma_top_theor)
            sigmas_bot_ratio.append(-var_array_y.values[1]/sigma_bot_theor)
            sigmas_diff_true.append(var_array_y.values[-2]-var_array_y.values[1])   # top - bottom (they're <0, so difference is >0)
            sigmas_diff_theor.append(sigma_bot_theor-sigma_top_theor)   # bottom - top (they're >0)
            sigmas_diff_ratio.append((var_array_y.values[-2]-var_array_y.values[1])/(sigma_bot_theor-sigma_top_theor))   # bottom - top (they're >0)
        # print(f'max Movement: {max(var_array_y.values)}')
    # end of dir loop
# ax3 = g_y.twinx()

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
    
    if has_res_directory:  # hue is res200 or res400
        g_x = sns.lineplot(data=df_x ,x="x",y=var_to_plot,ax=ax1,hue='res',style='rel_thr',alpha=0.5)
        g_y = sns.lineplot(data=df_y ,x="y",y=var_to_plot,ax=ax2,hue='res',style='rel_thr',alpha=0.5)
    else:  # hue is time
        g_x = sns.lineplot(data=df_x ,x="x",y=var_to_plot,ax=ax1,hue='time',style='scale',alpha=0.5)
        g_y = sns.lineplot(data=df_y ,x="y",y=var_to_plot,ax=ax2,hue='time',style='scale',alpha=0.5)
    # g_y = sns.lineplot(data=df_y ,x="y",y=var_to_plot,ax=ax2,hue='time',style='scale',alpha=0.5)
    # if var_to_plot == "Actual Movement" or var_to_plot == "Original Movement":
    #         ax1.axhline(y=relax_thresh[0], linestyle='--',color='red',label="first relaxation threshold")
    #         ax1.axhline(y=relax_thresh[1], linestyle='--',color='blue',label="second relaxation threshold")
    #         ax2.axhline(y=relax_thresh[0], linestyle='--',color='red')
    #         ax2.axhline(y=relax_thresh[1], linestyle='--',color='blue')

    if len(my_labels) == len(dir_list):
        handles, labels = ax1.get_legend_handles_labels()    # Extract the handles and labels from the existing legend    
        labels[-len(my_labels):] = my_labels             # Replace the labels for style
        ax1.legend(handles=handles, labels=labels,fancybox=True, ncol=1)
    else:
        ax1.legend(fancybox=True, ncol=1)   # legend for the horizontal line plot

    ax1.set_title("Horizontal Profile")
    #ax1.set_xlim([-0.51,+0.51])  # zoom in. Limits are location of max pressure +- 0.05
    #ax2.set_xlim([-1.01,+0.12])  # zoom in. Limits are location of max pressure +- 0.05
    ax2.set_title("Vertical Profile")

    g_y.legend_.remove()
    

    # save:
    #plt.savefig("gaussScale50-100-200_diff_hrz-vrt.png", dpi=600,transparent=True)
    plt.tight_layout()
    fig.suptitle(os.getcwd()) 
    plt.show()
if var_to_plot == "Sigma_1" or var_to_plot == "Sigma_2":
    if False:
        fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2)
        ax1.plot(sizes,sigmas_top_true,'-o',color='blue',label='top true')
        ax1.plot(sizes,sigmas_bot_true,'-o',color='green',label='bottom true')
        ax2.plot(sizes,sigmas_diff_true,'-o',label='difference true')
        ax1.plot(sizes,sigmas_top_theor,'--',color='blue',label='top theoretical')
        ax1.plot(sizes,sigmas_bot_theor,'--',color='green',label='bottom theoretical')
        ax2.plot(sizes,sigmas_diff_theor,'--',label='difference theoretical')
        

        set_axes_options(fig)
        ax1.set_ylabel(var_to_plot)
        plt.show()

    if False:  # plot ratios
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


