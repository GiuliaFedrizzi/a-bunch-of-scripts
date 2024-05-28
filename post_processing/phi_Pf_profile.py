"""
plot Pf, phi, k to see if they are correlated, how variable they are.
shade the area between Pf & phi and between 0 and k
"""
# import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import sys
sys.path.append('/home/home01/scgf/myscripts/post_processing')
from useful_functions import extract_two_profiles,getResolution



res = getResolution()    # resolution in the x direction
res_y = int(res*1.15)  # resolution in the y direction
filename = "my_experiment16000.csv"
dir_path = '/nobackup/scgf/myExperiments/threeAreas/prod/prt/prt45/rt0.5/visc_3_1e3/vis1e3_mR_08'

# define the indexes for the two profiles, the first one is the start for the horizontal, the second for the vertical profile
# REMEMBER to remove 1 from the horizontal index
point_indexes = [(res*res_y-(5*res))-1,5]   # top (horizontal) and left (vertical). X original: res*res_y/2+res, central: res*res_y/2 (?)
# point_indexes = [(res*res_y*0.75+res),5]   # 3/4 of the domain
# print(f'index for the horizontal profile: {point_indexes[0]}')


os.chdir(dir_path)
myfile = Path(os.getcwd()+'/'+filename)  # build file name including path


fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2)

vars_to_plot = ["Pressure","Porosity", "Permeability"]


# prepare to store vertical and horizontal data
all_data_v = {}
all_data_h = {}

for v in vars_to_plot:
    (x_v, y_v), (x_h, y_h) = extract_two_profiles(filename, v,point_indexes)
    all_data_v[v] = (x_v, y_v)
    all_data_h[v] = (x_h, y_h)
    
ax1a = ax1.twinx()
ax1b = ax1.twinx()
# print(np.corrcoef(abs(P_hor),poro_values_hor)) 

line1_h, = ax1.plot(all_data_h[vars_to_plot[0]][0], all_data_h[vars_to_plot[0]][1])
line2_h, = ax1a.plot(all_data_h[vars_to_plot[1]][0], all_data_h[vars_to_plot[1]][1],'g')  # plot porosity in green
line3_h, = ax1b.plot(all_data_h[vars_to_plot[2]][0], all_data_h[vars_to_plot[2]][1],color='lightgray',alpha=0.3)  # plot permeability
phi_scale_hor = max(all_data_h[vars_to_plot[0]][1])/max(all_data_h[vars_to_plot[1]][1])  # (max Pf)/(max phi) because if I want to plot them on the same y axis they have to be comparable
print(f'phi_scale_hor {phi_scale_hor}')
ax1.fill_between(all_data_h[vars_to_plot[0]][0], all_data_h[vars_to_plot[0]][1],all_data_h[vars_to_plot[1]][1]*phi_scale_hor,interpolate=True,alpha=0.2)  # x, y1, y2
ax1b.fill_between(all_data_h[vars_to_plot[2]][0], all_data_h[vars_to_plot[2]][1],y2=0,color='lightgray',alpha=0.3)  # fill between 0 and permeability 
ax1b.set_yscale('log')

# Legend
ax1.legend([line1_h, line2_h, line3_h], 
        [vars_to_plot[0], vars_to_plot[1],vars_to_plot[2]],loc=(0.6, 0.9))   # labels: y velocity and poro

ax1.set_xlabel('x') 
ax1.set_ylabel(vars_to_plot[0]) 
# ax1.set_ylim([6.6e7,1.09e8])  # Pf
# ax1a.set_ylim([0.13,0.32])    # phi
ax1a.set_ylabel(vars_to_plot[1], color='g')  # Setting color to match line color
ax1a.yaxis.tick_right()  # Ensure the y-axis label is on the right
ax1a.tick_params(axis='y', colors='g')

# Plotting on ax2
ax2a = ax2.twiny()
ax2b = ax2.twiny()

line1_v, = ax2.plot(all_data_v[vars_to_plot[0]][1], all_data_v[vars_to_plot[0]][0], label=vars_to_plot[0])
line2_v, = ax2a.plot(all_data_v[vars_to_plot[1]][1], all_data_v[vars_to_plot[1]][0],'g', label=vars_to_plot[1])
line3_v, = ax2b.plot(all_data_v[vars_to_plot[2]][1], all_data_v[vars_to_plot[2]][0],color='lightgray',alpha=0.3)

print(f'max Pf {max(all_data_v[vars_to_plot[0]][1])}')
print(f'max phi {max(all_data_v[vars_to_plot[1]][1])}')
phi_scale_ver = max(all_data_v[vars_to_plot[0]][1])/max(all_data_v[vars_to_plot[1]][1])  # (max Pf)/(max phi) because if I want to plot them on the same y axis they have to be comparable
print(f'phi_scale_ver {phi_scale_ver}')

ax2.fill_betweenx(all_data_v[vars_to_plot[1]][0], all_data_v[vars_to_plot[0]][1],all_data_v[vars_to_plot[1]][1]*phi_scale_ver,interpolate=True,alpha=0.2)  # y,x1,x2   between pressure and porosity 
ax2b.fill_betweenx(all_data_v[vars_to_plot[2]][0], all_data_v[vars_to_plot[2]][1],x2=0,color='lightgray',alpha=0.3)  # y,x1,x2    between 0 and permeability 
ax2b.set_xscale('log')

ax2.set_ylabel('y')
ax2.set_xlabel(vars_to_plot[0])
ax2a.set_xlabel(vars_to_plot[1], color='g')  # Setting color to match line color
ax2a.tick_params(axis='x', colors='g') 

# Legend
ax2.legend([line1_v, line2_v,line3_v], 
        [vars_to_plot[0], vars_to_plot[1],vars_to_plot[2]],loc=(0.8, 0.9))

fig.suptitle(str(myfile))
plt.show()

