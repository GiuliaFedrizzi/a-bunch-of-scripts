"""
plot Pf, phi, k to see if they are correlated, how variable they are.
version for the layers
Vertical profile only
"""
# import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import numpy as np
import os
from pathlib import Path
import sys
sys.path.append('/home/home01/scgf/myscripts/post_processing')
from useful_functions import extract_two_profiles,getResolution,extract_horiz_sum_of_bb

# two options: prt/prt46 (first paper, one layer) prt/layers (3rd paper, multiple layers)

prt_layers = False
prt46 = True

file_numbers = ["11000"] # for prt46 high visc
# file_numbers = ["08000","10000","14000","18000"] # for lr43/def0e-8/visc_3_1e3/z_vis1e3_mR_05
# file_numbers = ["07000","08000","10000","14000"]  # for lr43/def5e-7/visc_3_1e3/z_vis1e3_mR_05 and lr44/def5e-7/visc_3_1e3/vis1e3_mR_05
# dir_path = '/Users/giuliafedrizzi/Library/CloudStorage/OneDrive-UniversityofLeeds/PhD/arc/myExperiments/threeAreas/prod/prt/layers/lr43/def0e-8/visc_3_1e3/z_vis1e3_mR_05'
dir_path = '/Users/giuliafedrizzi/Library/CloudStorage/OneDrive-UniversityofLeeds/PhD/arc/myExperiments/threeAreas/prod/prt/prt46/rt0.5/visc_3_1e3/vis1e3_mR_08/'
# dir_path = '/nobackup/scgf/myExperiments/threeAreas/prod/prt/prt45/rt0.5/visc_1_1e15/vis1e15_mR_03'


# Pf_axes_lim = [6.6e7,1.6e8]  

if prt_layers:
    #  prt/layers
    Pf_axes_lim = [0.78,1.2]  # when scaled (pore fluid pressure, Pf0*0.5)
    phi_axes_lim = [0.09,0.22]
    k_axes_lim = [0,310]  # layer

if prt46:
    #  prt46
    Pf_axes_lim = [0.75,1.75]  # when scaled (pore fluid pressure, Pf0*0.5)
    phi_axes_lim = [0.11,0.42]
    k_axes_lim = [-1e-17,3.2e-16]  



os.chdir(dir_path)

res = getResolution()    # resolution in the x direction
res_y = int(res*1.15)  # resolution in the y direction
# define the indexes for the two profiles, the first one is the start for the horizontal, the second for the vertical profile
# REMEMBER to remove 1 from the horizontal index
point_indexes = [(res*res_y-(5*res))-1,5]   # top (horizontal) and left (vertical). X original: res*res_y/2+res, central: res*res_y/2 (?)
# point_indexes = [(res*res_y*0.75+res),5]   # 3/4 of the domain

blue_hex = '#1f77b4'  # colour for pressure

# extract initial profile
(_, y_press_v_0), (_, y_press_h_0) = extract_two_profiles("my_experiment00000.csv", "Pressure",point_indexes,res)


for filenum in file_numbers:
        filename = "my_experiment"+filenum+".csv"
        myfile = Path(os.getcwd()+'/'+filename)  # build file name including path

        plt.rcParams["font.weight"] = "bold"
        plt.rcParams['lines.linewidth'] = 3
        plt.rcParams['axes.linewidth'] = 3
        plt.rcParams['axes.labelsize'] = 20
        plt.rcParams['xtick.labelsize'] = 15
        plt.rcParams['ytick.labelsize'] = 15
        plt.rcParams['xtick.major.size'] = 5
        plt.rcParams['xtick.major.width'] = 3
        plt.rcParams['ytick.major.size'] = 5
        plt.rcParams['ytick.major.width'] = 3


        fig, (ax2,ax2a,ax2b) = plt.subplots(nrows=1, ncols=3,figsize=(6, 6), gridspec_kw={'wspace': 0.1})  # remove space between plots


        vars_to_plot = ["Pressure","Porosity", "Permeability","Broken Bonds"]


        # prepare to store vertical and horizontal data
        all_data_v = {}
        all_data_h = {}

        for v in vars_to_plot:
                (x_v, y_v), (x_h, y_h) = extract_two_profiles(filename, v,point_indexes,res)
                if v == "Pressure":   # normalise by initial value
                    y_v = y_v/y_press_v_0*0.8
                    y_h = y_h/y_press_h_0*0.8
                    
                all_data_v[v] = (x_v, y_v)
                all_data_h[v] = (x_h, y_h)
        

        # Plotting on ax2 -- vertical profile ---

        ax2.set_ylim([0.0,1])
        ax2a.set_ylim([0.0,1])
        ax2b.set_ylim([0.0,1])
        ax2.set_xlim(Pf_axes_lim)  # Pf
        ax2a.set_xlim(phi_axes_lim)  # phi
        ax2b.set_xlim(k_axes_lim)    # k - now it's bb sum
        # ax2.yaxis.tick_left()
        # ax2.yaxis.set_label_position("right")

        print(f'max Pf {max(all_data_v[vars_to_plot[0]][1])}')
        print(f'max phi {max(all_data_v[vars_to_plot[1]][1])}')

        if prt_layers:
            # sum of broken bonds along each horizontal line
            bb_horiz_sum,y_coord_bb_sum = extract_horiz_sum_of_bb(filename,res)
            line3_v, = ax2b.plot(bb_horiz_sum, y_coord_bb_sum,color='red',linewidth=2,alpha=0.7)
        line1_v, = ax2.plot(all_data_v[vars_to_plot[0]][1], all_data_v[vars_to_plot[0]][0], label=vars_to_plot[0])
        line2_v, = ax2a.plot(all_data_v[vars_to_plot[1]][1], all_data_v[vars_to_plot[1]][0],'g', label=vars_to_plot[1])
        if prt46:
            line3_v, = ax2b.plot(all_data_v[vars_to_plot[2]][1], all_data_v[vars_to_plot[2]][0],color='dimgray')
        
        #   add broken bonds
        y_coord_bb_ver, bb_values_ver = all_data_v[vars_to_plot[3]]
        x_coord_bb_hor, bb_values_hor = all_data_h[vars_to_plot[3]]

        bb_locations = [x for x, bb in zip(y_coord_bb_ver, bb_values_ver) if bb != 0]   # points of x_coord_vel_hor that have at least a broken bond

        # plot on the third axis (perm)
        ax2b.scatter(np.full((len(bb_locations),1), k_axes_lim[1]),bb_locations, color='red', marker='x', s=80)  # plot bb. array same length as bb_locations, full with the max range of Pf
        
        ax2.tick_params(axis='x', colors=blue_hex)
        # ax2.tick_params(axis='y')#,length=0) 
        # ax2.tick_params(axis='x') 
        ax2a.tick_params(axis='x', colors='g') 
        ax2b.tick_params(axis='x', colors='dimgray') 

        ax2.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax2.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax2a.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax2a.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax2b.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax2b.yaxis.set_major_locator(plt.MaxNLocator(4))
        if prt_layers:
            tick_spacing = 1/7
            ax2.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

        # hide numbers for poro and perm
        ax2a.set_yticklabels([])
        ax2b.set_yticklabels([])
        ax2.grid(True)
        ax2a.grid(True)
        ax2b.grid(True)

        if False:  # to create the legend for the axes (only 1 for all of the figures)
            # move all the ticks to the bottom
            ax2.xaxis.tick_bottom()
            ax2a.xaxis.tick_bottom()
            ax2b.xaxis.tick_bottom()
            ax2.spines['bottom'].set_position(('outward', 50)) 
            plt.setp(ax2.spines.values(), color=blue_hex)
            ax2b.spines['bottom'].set_position(('outward', 100))  # draw the axis for k further away from the axis for phi
            plt.setp(ax2b.spines.values(), color='lightgray')
            ax2a.spines['bottom'].set_position(('outward', 0)) 
            plt.setp(ax2a.spines.values(), color='g')
            ax2b.spines['left'].set_color('g')
            ax2b.spines['right'].set_color('g')
            ax2a.spines['bottom'].set_color('g')
            ax2.set_yticklabels([])
            ax2.tick_params(axis='y', which='both', length=0) # remove the ticks themselves

        if False:  # hide ticks
            ax2.set_yticklabels([])
            ax2.set_xticklabels([])
            ax2a.set_xticklabels([])
            ax2b.set_xticklabels([])
            ax2.spines[['right','top','bottom']].set_visible(False)
            ax2a.spines[['right','top','bottom']].set_visible(False)
            ax2b.spines[['right','top','bottom']].set_visible(False)
        #     ax2.tick_params(axis='y', which='both', length=0) # remove the ticks themselves
            ax2.tick_params(axis='x', which='both', length=0) # remove the ticks themselves
            ax2a.tick_params(axis='x', which='both', length=0) # remove the ticks themselves
            ax2b.tick_params(axis='x', which='both', length=0) # remove the ticks themselves

        ax2.spines[['right']].set_visible(False)
        ax2a.spines[['right']].set_visible(False)
        # ax2b.spines[['right']].set_visible(False)
        ax2b.spines['right'].set_linewidth(1)  # line where the broken bonds go. Thin but still visible
      
        fig_name = "phi_Pf0_0.8_3prof_"+str(filenum)+".png"
        plt.savefig(fig_name,dpi=600,transparent=True)
        # plt.show()

print("Done :)")
plt.show()

