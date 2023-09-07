""" 
plot stress (sigma 1) values at top and bottom from different simulations.
From diffusion_res_and_time_coord.py
Made for directories like /nobackup/scgf/myExperiments/gaussJan2022/gj78/
    which contain subdirs like size00200, ..., size10000


To be run from gaussJan2022
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
dir_labels = ['00200','00400','00600']#,'02000']#,'04000']#,'06000','08000']#,'10000']  # all sizes
# dir_labels = ['02000','04000','06000','08000','10000'] 
# gj_dirs = ['gj71','gj72','gj69','gj73','gj74','gj75','gj70','gj76']
# gj_lab = ['0.1','0.2','0.5','0.7','0.8','0.9','1','1.2'] # their labels

# gj_dirs = ['gj77','gj78','gj79','gj80','gj69','gj81','gj82']#,'gj76']
# gj_lab = ['1','1.2','1.5','1.8','2','2.2','2.5'] # their labels

# gj_dirs = ['gj60','gj61','gj62','gj63']
# gj_lab = ['100','200','400','800'] # their labels

# gj_dirs = ['gj87','gj88','gj89','gj90','gj91','gj86']
# gj_lab = ['0.1', '0.3', '0.5', '0.7', '0.9', '1'] # their labels

# gj_dirs = ['gj92','gj93','gj94','gj95']#,'gj91','gj86']#,'gj76']
# gj_lab = ['0.01', '0.03', '0.05', '0.07']#, '0.9', '1'] # their labels

# gj_dirs = ['gj97','gj98','gj99','gj100']#,'gj91','gj86']#,'gj76']
# gj_lab = ['0.01', '0.03', '0.05', '0.07'] # their labels

# gj_dirs = ['gj102','gj103','gj104','gj105']
# gj_lab = ['1000', '2000', '8000','9000']# their labels

# gj_dirs = ['gj138','gj140','gj139']
# gj_lab = ['3000','2700','1500']# densities

# gj_dirs = ['gj140','gj141']
# gj_lab = ['0','1']# only extension false / true

# gj_dirs = ['gj147','gj148']
# gj_lab = ['w (gj147)','w+fg (gj148)']# add weight to fg or not

# gj_dirs = ['gj149','gj150','gj151']
# gj_lab = ['1500','2700','3000']# densities

# gj_dirs = ['gj157','gj158','gj159']
# gj_lab = ['3000','2700','2500']# densities

# gj_dirs = ['gj159','gj160']
# gj_lab = ['gj159 (smaller relax)','gj160 (larger relax)']# densities

gj_dirs = ['sm13','sm14']
gj_lab = ['sm13 rho = 2700','sm14 rho = 3000']

gj_dirs = ['sm54','sm55']
gj_lab = ['sm54 h = 200','sm55 h = 800']

fig, (ax1,ax2,ax3) = plt.subplots(nrows=1,ncols=3)
resolution = 200

def dir_loop(gjdir,dir_labels,sigma_df,line_label):
    """ add data from every gj directory """
    
    first_file = 'my_experiment00500.csv'
    max_dir_size = float(dir_labels[-1])  # maximum size for the scaling of the plots: take the last directory (largest)
    for dir_label in dir_labels:
        """ loop through SIZE directories"""
        os.chdir('/nobackup/scgf/myExperiments/smooth/'+gjdir+'/size'+dir_label)
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
        # get density: it's the input parameter for setSolidDensity

        solid_density = getDensity()
        real_radius = myExp["real_radius"][0]
        sigma_1_top_theor = 0.66666 * solid_density*9.81*(float(depth)+2*real_radius)  # rho * g * (depth + first row)
        sigma_1_bot_theor = 0.66666 * solid_density*9.81*(float(depth)+float(dom_size))  # rho * g * (depth+size)

        # build the dataframe
        df_temp = pd.DataFrame({
            'top true': [-var_array_y.values[-2]],
            'top theor': [sigma_1_top_theor],
            'top ratio': [-var_array_y.values[-2]/sigma_1_top_theor],
            'bottom true': [-var_array_y.values[0]],
            'bottom theor': [sigma_1_bot_theor],
            'bottom ratio': [-var_array_y.values[0]/sigma_1_bot_theor],
            'difference true': [var_array_y.values[-2]-var_array_y.values[0]],
            'difference theor': [sigma_1_bot_theor-sigma_1_top_theor],
            'difference ratio': [(var_array_y.values[-2]-var_array_y.values[0])/(sigma_1_bot_theor-sigma_1_top_theor)],
            'scale': [float(dir_label)],
            'gj_dir': [gjdir],
            'coeff': [line_label]}
            )  # temporary dataframe (to be merged with global df)

        sigma_df = pd.concat([sigma_df,df_temp], ignore_index=True) # append to old one
    return sigma_df

sigma_df = pd.DataFrame()
print('\n')
for i,gjdir in enumerate(gj_dirs):
    line_label = gj_lab[i]  # the coefficient used to calculate gravity or weight
    print(line_label)
    sigma_df = dir_loop(gjdir,dir_labels,sigma_df,line_label)

if True:
    # plot both true and theoretical=simulated values (2 lines each plot)
    # ts = value at the top, simulated. tt = value at the top, theoretical
    g_ts = sns.lineplot(data=sigma_df ,x="scale",y='top true',ax=ax1,hue='coeff',markers=True,marker='o',alpha=0.5)
    g_tt = sns.lineplot(data=sigma_df ,x="scale",y='top theor',linestyle='--',ax=ax1,hue='coeff',alpha=0.5)#,dashes=[(2, 2), (2, 2)],alpha=0.5)
    g_bs = sns.lineplot(data=sigma_df ,x="scale",y='bottom true',ax=ax2,hue='coeff',markers=True,marker='o',alpha=0.5)
    g_bt = sns.lineplot(data=sigma_df ,x="scale",y='bottom theor',linestyle='--',ax=ax2,hue='coeff',alpha=0.5)#,dashes=[(2, 2), (2, 2)],alpha=0.5)
    g_ds = sns.lineplot(data=sigma_df ,x="scale",y='difference true',ax=ax3,hue='coeff',marker='o',alpha=0.5)#,dashes=[(2, 2), (2, 2)],alpha=0.5)
    g_dt = sns.lineplot(data=sigma_df ,x="scale",y='difference theor',linestyle='--',ax=ax3,hue='coeff',alpha=0.5)#,dashes=[(2, 2), (2, 2)],alpha=0.5)
if False:
    #  g_tr = top ratio
    g_tr = sns.lineplot(data=sigma_df ,x="scale",y='top ratio',ax=ax1,hue='coeff',markers=True,marker='o',alpha=0.5)
    ax1.axhline(y=1, color='k', linestyle='--',alpha=0.3)
    g_br = sns.lineplot(data=sigma_df ,x="scale",y='bottom ratio',ax=ax2,hue='coeff',markers=True,marker='o',alpha=0.5)
    ax2.axhline(y=1, color='k', linestyle='--',alpha=0.3)
    g_br = sns.lineplot(data=sigma_df ,x="scale",y='difference ratio',ax=ax3,hue='coeff',markers=True,marker='o',alpha=0.5)
    ax3.axhline(y=1, color='k', linestyle='--',alpha=0.3)
    
for ax in fig.axes:
    ax.set_xticks(ticks=[float(x) for x in dir_labels])  # define list of ticks

ax1.set_title("Top value")
ax2.set_title("Bottom value")
ax3.set_title("Difference")
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
