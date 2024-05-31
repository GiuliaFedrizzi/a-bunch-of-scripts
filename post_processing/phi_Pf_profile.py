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


file_numbers = ["01000","03000","07000","16000"]
# dir_path = '/nobackup/scgf/myExperiments/threeAreas/prod/prt/prt45/rt0.5/visc_1_1e15/vis1e15_mR_03'
dir_path = '/nobackup/scgf/myExperiments/threeAreas/prod/prt/prt45/rt0.5/visc_3_1e3/vis1e3_mR_08'


# Pf_axes_lim = [6.6e7,1.1e8]  # case A  
Pf_axes_lim = [6.6e7,1.6e8]  
# phi_axes_lim = [0.134,0.30]  # case A
phi_axes_lim = [0.10,0.42]
# k_axes_lim = [7.22131e-19,1.66541e-16]  # case A
k_axes_lim = [-1e-18,3.2e-16]

label_size = 13

os.chdir(dir_path)

res = getResolution()    # resolution in the x direction
res_y = int(res*1.15)  # resolution in the y direction
# define the indexes for the two profiles, the first one is the start for the horizontal, the second for the vertical profile
# REMEMBER to remove 1 from the horizontal index
point_indexes = [(res*res_y-(5*res))-1,5]   # top (horizontal) and left (vertical). X original: res*res_y/2+res, central: res*res_y/2 (?)
# point_indexes = [(res*res_y*0.75+res),5]   # 3/4 of the domain
# print(f'index for the horizontal profile: {point_indexes[0]}')


for filenum in file_numbers:
        filename = "my_experiment"+filenum+".csv"
        myfile = Path(os.getcwd()+'/'+filename)  # build file name including path


        plt.rcParams["font.weight"] = "bold"
        plt.rcParams['lines.linewidth'] = 4
        plt.rcParams['axes.linewidth'] = 2
        plt.rcParams['axes.labelsize'] = 20
        plt.rcParams['xtick.labelsize'] = 18
        plt.rcParams['ytick.labelsize'] = 18

        fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2,figsize=(17, 9))
        plt.setp(ax1.spines.values(), linewidth=3)  # #f56c42
        plt.setp(ax2.spines.values(), linewidth=3)  # #9ce35d

        vars_to_plot = ["Pressure","Porosity", "Permeability"]


        # prepare to store vertical and horizontal data
        all_data_v = {}
        all_data_h = {}

        for v in vars_to_plot:
                (x_v, y_v), (x_h, y_h) = extract_two_profiles(filename, v,point_indexes,res)
                all_data_v[v] = (x_v, y_v)
                all_data_h[v] = (x_h, y_h)
        
        ax1a = ax1.twinx()
        ax1b = ax1.twinx()
        # print(np.corrcoef(abs(P_hor),poro_values_hor)) 

        line1_h, = ax1.plot(all_data_h[vars_to_plot[0]][0], all_data_h[vars_to_plot[0]][1])
        line2_h, = ax1a.plot(all_data_h[vars_to_plot[1]][0], all_data_h[vars_to_plot[1]][1],'g')  # plot porosity in green
        line3_h, = ax1b.plot(all_data_h[vars_to_plot[2]][0], all_data_h[vars_to_plot[2]][1],color='lightgray',alpha=0.3)  # plot permeability

        ax1b.fill_between(all_data_h[vars_to_plot[2]][0], all_data_h[vars_to_plot[2]][1],y2=0,color='lightgray',alpha=0.3)  # fill between 0 and permeability 
        # ax1b.set_yscale('log')

        # Legend
        # ax1.legend([line1_h, line2_h, line3_h], 
                # [vars_to_plot[0], vars_to_plot[1],vars_to_plot[2]],loc=(0.6, 0.8))   # labels: y velocity and poro

        #   limits:
        # ax1.set_ylabel(vars_to_plot[0]) 
        ax1.set_ylim(Pf_axes_lim)  # Pf
        ax1a.set_ylim(phi_axes_lim)    # phi
        ax1b.set_ylim(k_axes_lim)    # k
        ax1.set_xlim([0,1])
        # ax1a.set_ylabel(vars_to_plot[1], color='g')  # Setting color to match line color
        # ax1.set_xlabel('x') 

        #   options:
        ax1a.yaxis.tick_right()  # Ensure the y-axis label is on the right
        ax1.tick_params(axis='x')
        ax1.tick_params(axis='y', colors='#1f77b4')
        ax1.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax1.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax1a.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax1b.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax1a.tick_params(axis='y', colors='g')
        ax1b.tick_params(axis='y', colors='darkgray') 
        ax1b.spines['right'].set_position(('outward', 60))  # draw the axis for k further from the axis for phi



        # Plotting on ax2 -- vertical profile ---
        ax2a = ax2.twiny()
        ax2b = ax2.twiny()

        ax2.set_ylim([0,0.99])
        ax2.set_xlim(Pf_axes_lim)  # Pf
        ax2a.set_xlim(phi_axes_lim)  # phi
        ax2b.set_xlim(k_axes_lim)    # k
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position("right")

        print(f'max Pf {max(all_data_v[vars_to_plot[0]][1])/1e8}e8')
        print(f'max phi {max(all_data_v[vars_to_plot[1]][1])}')

        line1_v, = ax2.plot(all_data_v[vars_to_plot[0]][1], all_data_v[vars_to_plot[0]][0], label=vars_to_plot[0])
        line2_v, = ax2a.plot(all_data_v[vars_to_plot[1]][1], all_data_v[vars_to_plot[1]][0],'g', label=vars_to_plot[1])
        line3_v, = ax2b.plot(all_data_v[vars_to_plot[2]][1], all_data_v[vars_to_plot[2]][0],color='lightgray',alpha=0.3)

        ax2b.fill_betweenx(all_data_v[vars_to_plot[2]][0], all_data_v[vars_to_plot[2]][1],x2=0,color='lightgray',alpha=0.3)  # y,x1,x2    between 0 and permeability 
        # ax2b.set_xscale('log')
        ax2b.spines['top'].set_position(('outward', 30))  # draw the axis for k further from the axis for phi

        # ax2.set_ylabel('y')
        # ax2.set_xlabel(vars_to_plot[0])
        # ax2a.set_xlabel(vars_to_plot[1], color='g')  # Setting color to match line color
        ax2.tick_params(axis='x', colors='#1f77b4')
        ax2.tick_params(axis='y') 
        ax2.tick_params(axis='x') 
        ax2a.tick_params(axis='x', colors='g') 
        ax2b.tick_params(axis='x', colors='darkgray') 
        ax2.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax2.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax2a.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax2b.xaxis.set_major_locator(plt.MaxNLocator(4))
        # Legend
        # ax2.legend([line1_v, line2_v,line3_v], 
                # [vars_to_plot[0], vars_to_plot[1],vars_to_plot[2]],loc=(0.7, 0.8))

        fig.tight_layout()  # must go before fill_between

        # fill between phi and Pf
        # transform datasets to display space
        # WARNING it only works if I set the limits for the axes manually
        x1p, y1p = ax1.transData.transform(np.c_[all_data_h[vars_to_plot[0]][0],all_data_h[vars_to_plot[0]][1]]).T
        _, y2p = ax1a.transData.transform(np.c_[all_data_h[vars_to_plot[1]][0],all_data_h[vars_to_plot[1]][1]]).T

        ax1.autoscale(False)
        ax1.fill_between(x1p, y1p, y2p, color='teal',alpha=0.2, transform=None)
        
        x1p, yp = ax2.transData.transform(np.c_[all_data_v[vars_to_plot[0]][1],all_data_v[vars_to_plot[0]][0]]).T
        x2p, _ = ax2a.transData.transform(np.c_[all_data_v[vars_to_plot[1]][1],all_data_v[vars_to_plot[1]][0]]).T
        ax2.autoscale(False)
        ax2.fill_betweenx(yp, x1p, x2p, color="teal", alpha=0.2, transform=None)

        # print(f'horizontal min {min(all_data_h[vars_to_plot[2]][1])}, max {max(all_data_h[vars_to_plot[2]][1])}, \nvertical min {min(all_data_v[vars_to_plot[2]][1])}, max {max(all_data_v[vars_to_plot[2]][1])}')

        # fig.suptitle(str(myfile))
        plt.savefig("phi_Pf_"+str(filenum)+".png")
print("Done :)")
# plt.show()

