""" 
plot stress (sigma 1) values at top and bottom from different simulations.
From diffusion_res_and_time_coord.py
Made for directories like /nobackup/scgf/myExperiments/gaussJan2022/gj78/
    which contain subdirs like size00200, ..., size10000
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import glob
import seaborn as sns
from pathlib import Path

# import functions from external file
from useful_functions import * 

# dir_labels = ['00200', '00400','00600','00800','01000']
dir_labels = ['00200', '00400','00600','00800','01000','02000','04000','06000','08000','10000']  # all sizes
# dir_labels = ['02000','04000','06000','08000','10000'] 
gj_dirs = ['gj77','gj78']

fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2)
resolution = 200

def dir_loop(gjdir,dir_labels,sigma_df):
    """ add data from every gj directory """
    
    first_file = 'my_experiment00100.csv'
    max_dir_size = float(dir_labels[-1])  # maximum size for the scaling of the plots: take the last directory (largest)
    for dir_label in dir_labels:
        """ loop through directories"""
        os.chdir('/nobackup/scgf/myExperiments/gaussJan2022/'+gjdir+'/size'+dir_label)
        scale_factor = float(dir_label)/max_dir_size # factor for scaling the axes. Normalised by the maximum size (e.g. 1000)

        # open the first file after melt addition to find the max P
        if not os.path.exists(os.getcwd()+'/'+first_file):  # if it can't find the file after my_experiment00000.csv
            print('skipping '+os.getcwd())
            continue
        myExp = pd.read_csv(first_file, header=0)
        if dir_label == '10000':
            xmax = myExp.loc[45707, 'x coord']
            ymax = myExp.loc[45707, 'y coord']
            max_id = 45707   #  index of the point with the maximum pressure value
            offset = 100   # so that they are all centered around 0 on the x axis. It's the shift in the x direction.
        else:        
            xmax = myExp.loc[myExp['Pressure'].idxmax(), 'x coord']
            ymax = myExp.loc[myExp['Pressure'].idxmax(), 'y coord']
            max_id = myExp['Pressure'].idxmax()   #  index of the point with the maximum pressure value
            offset = max_id%resolution   # so that they are all centered around 0 on the x axis. It's the shift in the x direction.
        
        # find the x coordinates that correspond to the max pressure, based on their index
        x_array = myExp.iloc[(max_id-offset):(max_id-offset)+resolution:1,myExp.columns.get_loc('x coord')] 
        y_array = myExp.iloc[offset::resolution,myExp.columns.get_loc('y coord')] # first: the point with coorinate = offset. Then every point above it (with a period of 'resolution') 
        
        var_array_x =  myExp.iloc[(max_id-offset):(max_id-offset)+resolution:1,myExp.columns.get_loc('Sigma_1')] 
        var_array_y = myExp.iloc[offset::resolution,myExp.columns.get_loc('Sigma_1')]
        
        dom_size = float(getParameterFromLatte('input.txt','Scale'))
        depth = getDepth("input.txt")
        sigma_1_top_theor = 2968.94*9.81*float(depth)  # rho * g * depth
        sigma_1_bot_theor = 2968.94*9.81*(float(depth)+float(dom_size))  # rho * g * (depth+size)

        df_temp = pd.DataFrame({
            'top true': [-var_array_y.values[-2]],
            'top theor': [sigma_1_top_theor],
            'top ratio': [-var_array_y.values[-2]/sigma_1_top_theor],
            'bottom true': [-var_array_y.values[0]],
            'bottom theor': [sigma_1_bot_theor],
            'bottom ratio': [-var_array_y.values[0]/sigma_1_bot_theor],
            'scale': [float(dir_label)],
            'gj_dir': [gjdir]}
            )  # temporary dataframe (to be merged with global df)

        sigma_df = pd.concat([sigma_df,df_temp], ignore_index=True) # append to old one
    return sigma_df

sigma_df = pd.DataFrame()
for gjdir in gj_dirs:
    sigma_df = dir_loop(gjdir,dir_labels,sigma_df)

if False:
    # ts = value at the top, simulated. tt = value at the top, theoretical
    g_ts = sns.lineplot(data=sigma_df ,x="scale",y='top true',ax=ax1,hue='gj_dir',markers=True,marker='o',alpha=0.5)
    g_tt = sns.lineplot(data=sigma_df ,x="scale",y='top theor',ax=ax1,hue='gj_dir',alpha=0.5)#,dashes=[(2, 2), (2, 2)],alpha=0.5)
    g_bs = sns.lineplot(data=sigma_df ,x="scale",y='bottom true',ax=ax2,hue='gj_dir',markers=True,marker='o',alpha=0.5)
    g_bt = sns.lineplot(data=sigma_df ,x="scale",y='bottom theor',ax=ax2,hue='gj_dir',alpha=0.5)#,dashes=[(2, 2), (2, 2)],alpha=0.5)
if True:
    #  g_tr = top ratio
    g_tr = sns.lineplot(data=sigma_df ,x="scale",y='top ratio',ax=ax1,hue='gj_dir',markers=True,marker='o',alpha=0.5)
    ax1.axhline(y=1, color='k', linestyle='--',alpha=0.3)
    g_br = sns.lineplot(data=sigma_df ,x="scale",y='bottom ratio',ax=ax2,hue='gj_dir',markers=True,marker='o',alpha=0.5)
    ax2.axhline(y=1, color='k', linestyle='--',alpha=0.3)
ax1.set_title("Top value")
ax2.set_title("Bottom value")
plt.show()

# if False:
#     ax1.plot(sizes,sigmas_top_true,'-o',label='top true')
#     ax2.plot(sizes,sigmas_bot_true,'-o',label='bottom true')
#     ax1.plot(sizes,sigmas_top_theor,'--',label='top theoretical')
#     ax2.plot(sizes,sigmas_bot_theor,'--',label='bottom theoretical')
#     ax1.legend()
#     ax2.legend()
#     os.chdir('..')
#     fig.suptitle(os.getcwd()) # get the part of the path that is after "myExperiments/"
#     plt.show()
