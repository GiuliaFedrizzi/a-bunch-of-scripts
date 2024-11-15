"""
plot Pf, phi, k to see if they are correlated, how variable they are.
shade the area between Pf & phi and between 0 and k
version for the layers
"""
# import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
from pathlib import Path
import sys
sys.path.append('/home/home01/scgf/myscripts/post_processing')
from useful_functions import extract_two_profiles,getResolution,extract_horiz_sum_of_bb


# file_numbers = ["10000"] 
file_numbers = ["08000","10000","14000","18000"] # for lr43/def0e-8/visc_3_1e3/z_vis1e3_mR_05
# file_numbers = ["07000","08000","10000","14000"]  # for lr43/def5e-7/visc_3_1e3/z_vis1e3_mR_05 and lr44/def5e-7/visc_3_1e3/vis1e3_mR_05
dir_path = '/nobackup/scgf/myExperiments/threeAreas/prod/prt/layers/lr43/def0e-8/visc_3_1e3/z_vis1e3_mR_05'
# dir_path = '/nobackup/scgf/myExperiments/threeAreas/prod/prt/prt45/rt0.5/visc_1_1e15/vis1e15_mR_03'


# Pf_axes_lim = [6.6e7,1.6e8]  
Pf_axes_lim = [0.78,1.2]  # when scaled (pore fluid pressure, Pf0*0.5)
phi_axes_lim = [0.09,0.22]
k_axes_lim = [0,310]  # layer


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
        plt.rcParams['lines.linewidth'] = 4
        plt.rcParams['axes.linewidth'] = 4
        plt.rcParams['axes.labelsize'] = 20
        plt.rcParams['xtick.labelsize'] = 30
        plt.rcParams['ytick.labelsize'] = 30
        plt.rcParams['xtick.major.size'] = 10
        plt.rcParams['xtick.major.width'] = 6
        plt.rcParams['ytick.major.size'] = 10
        plt.rcParams['ytick.major.width'] = 6


        fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2,figsize=(12, 9))
        # plt.setp(ax1.spines.values(), linewidth=3)  # #f56c42
        # plt.setp(ax2.spines.values(), linewidth=3)  # #9ce35d

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
        
        ax1a = ax1.twinx()
        ax1b = ax1.twinx()
        # print(np.corrcoef(abs(P_hor),poro_values_hor)) 

        line1_h, = ax1.plot(all_data_h[vars_to_plot[0]][0], all_data_h[vars_to_plot[0]][1])
        line2_h, = ax1a.plot(all_data_h[vars_to_plot[1]][0], all_data_h[vars_to_plot[1]][1],'g')  # plot porosity in green
        line3_h, = ax1b.plot(all_data_h[vars_to_plot[2]][0], all_data_h[vars_to_plot[2]][1],color='lightgray',alpha=0.3)  # plot permeability

        ax1b.fill_between(all_data_h[vars_to_plot[2]][0], all_data_h[vars_to_plot[2]][1],y2=0,color='lightgray',alpha=0.3)  # fill between 0 and permeability 
        


        # Legend
        # ax1.legend([line1_h, line2_h, line3_h], 
                # [vars_to_plot[0], vars_to_plot[1],vars_to_plot[2]],loc=(0.6, 0.8))   # labels: y velocity and poro

        #   add broken bonds
        x_coord_bb_hor, bb_values_hor = all_data_h[vars_to_plot[3]]

        bb_locations = [x for x, bb in zip(x_coord_bb_hor, bb_values_hor) if bb != 0]   # points of x_coord_vel_hor that have at least a broken bond
        ax1.scatter(bb_locations, np.full((len(bb_locations),1), Pf_axes_lim[1]), color='red', marker='x', s=150)  # plot bb. array same length as bb_locations, full with the max range of Pf
        
        
        #   limits:
        # ax1.set_ylabel(vars_to_plot[0]) 
        ax1.set_ylim(Pf_axes_lim)  # Pf
        ax1a.set_ylim(phi_axes_lim)    # phi
        ax1b.set_ylim(k_axes_lim)    # k
        ax1.set_xlim([0,1])
        # ax1a.set_ylabel(vars_to_plot[1], color='g')  # Setting color to match line color
        # ax1.set_xlabel('x') 

        #   options:
        ax1a.yaxis.tick_right()  # Ensure the y-axis label is on the right (phi)
        ax1b.yaxis.tick_right()  # Ensure the y-axis label is on the left (k)
        ax1.tick_params(axis='x')
        ax1.tick_params(axis='y', colors=blue_hex) # blue
        ax1.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax1.yaxis.set_major_locator(plt.MaxNLocator(3))
        ax1a.yaxis.set_major_locator(plt.MaxNLocator(3))
        ax1a.tick_params(axis='y', colors='g')
        # ax1b.tick_params(axis='y', colors='darkgray') 
        ax1b.spines['left'].set_position(('outward', 10))  # draw the axis for k further from the axis for phi
        if True:  # hide ticks
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
            ax1a.set_yticklabels([])
            ax1.tick_params(axis='y', which='both', length=0) # remove the ticks themselves
            ax1a.tick_params(axis='y', which='both', length=0) # remove the ticks themselves


        # Plotting on ax2 -- vertical profile ---
        ax2a = ax2.twiny()
        ax2b = ax2.twiny()

        ax2.set_ylim([0,0.99])
        ax2.set_xlim(Pf_axes_lim)  # Pf
        ax2a.set_xlim(phi_axes_lim)  # phi
        ax2b.set_xlim(k_axes_lim)    # k - now it's bb sum
        ax2.yaxis.tick_left()
        # ax2.yaxis.set_label_position("right")

        print(f'max Pf {max(all_data_v[vars_to_plot[0]][1])}')
        print(f'max phi {max(all_data_v[vars_to_plot[1]][1])}')

        # sum of broken bonds along each horizontal line
        bb_horiz_sum,y_coord_bb_sum = extract_horiz_sum_of_bb(filename,res)
        line3_v, = ax2b.plot(bb_horiz_sum, y_coord_bb_sum,color='red',linewidth=2,alpha=0.7)
        line1_v, = ax2.plot(all_data_v[vars_to_plot[0]][1], all_data_v[vars_to_plot[0]][0], label=vars_to_plot[0])
        line2_v, = ax2a.plot(all_data_v[vars_to_plot[1]][1], all_data_v[vars_to_plot[1]][0],'g', label=vars_to_plot[1])
        # line3_v, = ax2b.plot(all_data_v[vars_to_plot[2]][1], all_data_v[vars_to_plot[2]][0],color='lightgray',alpha=0.3)
        
        
        ax2.tick_params(axis='x', colors=blue_hex)
        ax2.tick_params(axis='y') 
        ax2.tick_params(axis='x') 
        ax2a.tick_params(axis='x', colors='g') 
        ax2b.tick_params(axis='x', colors='darkgray') 
        ax2.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax2.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax2a.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax2b.xaxis.set_major_locator(plt.MaxNLocator(3))

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

        if True:  # hide ticks
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

        # add shading to show where layers are
        # ax2.fill_between(np.arange(Pf_axes_lim[0],Pf_axes_lim[1]+0.06,0.05), y1=1.0/7.0, y2=2.0/7.0, color="orange",interpolate=True, alpha=0.2)#, transform=None)
        # ax2.fill_between(np.arange(Pf_axes_lim[0],Pf_axes_lim[1]+0.06,0.05), y1=3.0/7.0, y2=4.0/7.0, color="orange",interpolate=True, alpha=0.2)#, transform=None)
        # ax2.fill_between(np.arange(Pf_axes_lim[0],Pf_axes_lim[1]+0.06,0.05), y1=5.0/7.0, y2=6.0/7.0, color="orange",interpolate=True, alpha=0.2)#, transform=None)
        # fig.suptitle(str(myfile))
        fig_name = "phi_Pf0_0.8_"+str(filenum)+".png"
        # plt.tight_layout()
        plt.savefig(fig_name)
        # plt.show()
        # plt.savefig("phi_Pf_vert_ticks.png")
        # break

print("Done :)")
plt.show()

