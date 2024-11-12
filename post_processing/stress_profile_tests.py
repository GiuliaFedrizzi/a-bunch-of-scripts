"""
plot stress profile for the tests chapter
to show that res200 and res400 are equivalent 
"""
# import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
from pathlib import Path
import glob
import sys

sys.path.append('/home/home01/scgf/myscripts/post_processing')
from useful_functions import extract_two_profiles,getResolution


res_dirs = ['res200','res400']
dir_path = '/nobackup/scgf/myExperiments/threeAreas/prod/prt/singleInjection/si02'

Pf_axes_lim = [1,2.1]  # when scaled (pore fluid pressure)
Pf_axes_lim = [0.78,2]  # when scaled (pore fluid pressure, Pf0*0.5)
phi_axes_lim = [0.11,0.42]
# phi_axes_lim = [0.10,0.45]  # layer
k_axes_lim = [-1e-18,3.2e-16]


os.chdir(dir_path)

blue_hex = '#1f77b4'  # colour for pressure

fig, (ax1) = plt.subplots(nrows=1, ncols=1,figsize=(9, 9))

plt.rcParams["font.weight"] = "bold"
plt.rcParams['lines.linewidth'] = 2.5
plt.rcParams['axes.linewidth'] = 2

filename = "my_experiment00000.csv"
plt.setp(ax1.spines.values(), linewidth=3)  # #f56c42

colours = ['#E7AD99','#CE796B','#89AAE6',  '#0471A6',   '#061826'] 
         # Melon, Old rose, Vista Blue, Honolulu Blue, Rich black
        # not using:  Vista Blue '#89AAE6'
        # not using:  Blue (NCS)  3685B5
col_counter = 0

for res_dir in res_dirs:
    os.chdir(res_dir)
#     myfile = Path(os.getcwd()+"/"+rt_dir+'/'+filename)  # build file name including path
    res = getResolution()    # resolution in the x direction
    res_y = int(res*1.15)  # resolution in the y direction
    
    rt_dirs = sorted(glob.glob("rb0*"))

    # define the indexes for the two profiles, the first one is the start for the horizontal, the second for the vertical profile
    # REMEMBER to remove 1 from the horizontal index
    # point_indexes = [(res*res_y*0.75+res),5]   # 3/4 of the domain
    # print(f'index for the horizontal profile: {point_indexes[0]}')
    point_indexes = [(res*res_y-(5*res))-1,5]   # top (horizontal) and left (vertical). X original: res*res_y/2+res, central: res*res_y/2 (?)

    for rt_dir in rt_dirs:   
        # print(f'rt_dir {rt_dir}') 
        if res_dir == "res400" and (rt_dir == "rb0.03" or rt_dir == "rb0.01" or rt_dir == "rb0.007"):  # skip these values of res400
            continue
        os.chdir(rt_dir)
        
        # _v  because it's the vertical profile
        (coords_v, stress_v), _ = extract_two_profiles(filename, "Sigma_1",point_indexes,res)
        stress_v = -stress_v/1e6
        
        # Plotting on ax1 -- vertical profile ---

        ax1.set_ylim([0,1.0])
        if res_dir == "res200":
            alpha_value = 1.0
            res_label = "200"
        else:
            alpha_value = 0.8
            res_label = "400"
        rt_label = rt_dir.split("b")[1]  # take what comes after "b" = 0.01, 0.03 etc
        line1_v, = ax1.plot(stress_v,coords_v,color=colours[col_counter],alpha=alpha_value, label="Resolution in x: "+res_label+", r$_t$ = "+rt_label)
        col_counter+=1

        # Legend
        ax1.legend() #loc=(0.7, 0.8))

        #   options:
        ax1.tick_params(axis='x')
        ax1.tick_params(axis='y')#, colors=blue_hex) # blue
        ax1.xaxis.set_major_locator(plt.MaxNLocator(4))
        fig.tight_layout() 
        ax1.set_xlabel('$\sigma_1$ (MPa)',fontsize=15) 
        ax1.set_ylabel("y coordinate",fontsize=15) 
        ax1.tick_params(width=2, length=4)


        print(f'max {max(stress_v)}, min {min(stress_v)}, diff {max(stress_v)-min(stress_v)}')
        os.chdir("..")
    os.chdir("..")
        
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid(linestyle = '--', linewidth = 0.5)

print("Done :)")
# plt.show()
fig_name = "si02_sigma_1_res200_400.png"
plt.savefig(fig_name,dpi=600)

