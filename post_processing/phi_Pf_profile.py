"""
plot Pf, phi, k to see if they are correlated, how variable they are.

"""
# import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
from pathlib import Path
import sys
sys.path.append('/home/home01/scgf/myscripts/post_processing')
from useful_functions import extract_two_profiles,getResolution


file_numbers = ["01000","03000","07000","16000"]  # tsteps for visc_3_1e3/vis1e3_mR_08
# file_numbers = ["30000","50000","80000","100000"]  # tsteps for visc_1_1e15/vis1e15_mR_03
dir_path = '/Users/giuliafedrizzi/Library/CloudStorage/OneDrive-UniversityofLeeds/PhD/arc/myExperiments/threeAreas/prod/prt/prt45/rt0.5/visc_3_1e3/vis1e3_mR_08'
# dir_path = '/Users/giuliafedrizzi/Library/CloudStorage/OneDrive-UniversityofLeeds/PhD/arc/myExperiments/threeAreas/prod/prt/prt45/rt0.5/visc_1_1e15/vis1e15_mR_03'

k_log = True

# Pf_axes_lim = [6.6e7,1.6e8]  
# Pf_axes_lim = [1,2.1]  # when scaled (pore fluid pressure)
Pf_axes_lim = [0.75,2]  # when scaled (pore fluid pressure, Pf0*0.5)
phi_axes_lim = [0.11,0.42]
# phi_axes_lim = [0.10,0.45]  # layer
if k_log:

        k_axes_lim = [0,3.2e-16]
else:
        k_axes_lim = [-1e-17,3.2e-16]


os.chdir(dir_path)

res = getResolution()    # resolution in the x direction
res_y = int(res*1.15)  # resolution in the y direction
# define the indexes for the two profiles, the first one is the start for the horizontal, the second for the vertical profile
# REMEMBER to remove 1 from the horizontal index
point_indexes = [(res*res_y-(5*res))-1,5]   # top (horizontal) and left (vertical). X original: res*res_y/2+res, central: res*res_y/2 (?)
# point_indexes = [(res*res_y*0.75+res),5]   # 3/4 of the domain
# print(f'index for the horizontal profile: {point_indexes[0]}')

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
        if k_log:
                plt.rcParams['ytick.labelsize'] = 15
                plt.rcParams['ytick.minor.size'] = 5
                plt.rcParams['ytick.minor.width'] = 3
        plt.rcParams['xtick.major.size'] = 6
        plt.rcParams['xtick.major.width'] = 4
        plt.rcParams['ytick.major.size'] = 6
        plt.rcParams['ytick.major.width'] = 4


        fig, (ax1,ax1a,ax1b) = plt.subplots(nrows=3, ncols=1,figsize=(6, 6), gridspec_kw={'hspace': 0})  # remove space between plots
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
        if k_log:  # needs to go before I plot anything
                ax1b.set_yscale('log')
        # print(np.corrcoef(abs(P_hor),poro_values_hor)) 

        line1_h, = ax1.plot(all_data_h[vars_to_plot[0]][0], all_data_h[vars_to_plot[0]][1])
        line2_h, = ax1a.plot(all_data_h[vars_to_plot[1]][0], all_data_h[vars_to_plot[1]][1],'g')  # plot porosity in green
        line3_h, = ax1b.plot(all_data_h[vars_to_plot[2]][0], all_data_h[vars_to_plot[2]][1],color='dimgray')  # plot permeability

        # ax1b.fill_between(all_data_h[vars_to_plot[2]][0], all_data_h[vars_to_plot[2]][1],y2=0,color='lightgray',alpha=0.3)  # fill between 0 and permeability 
        


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
        ax1a.set_xlim([0,1])
        ax1b.set_xlim([0,1])
        # ax1a.set_ylabel(vars_to_plot[1], color='g')  # Setting color to match line color
        # ax1.set_xlabel('x') 

        #   options:
        ax1.yaxis.tick_right()  # Ensure the y-axis label is on the right (Pf)
        ax1a.yaxis.tick_right()  # Ensure the y-axis label is on the right (phi)
        ax1b.yaxis.tick_right()  # Ensure the y-axis label is on the right (k)
        ax1.tick_params(axis='x')
        ax1.tick_params(axis='y', colors=blue_hex) # blue
        ax1.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax1a.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax1b.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax1.yaxis.set_major_locator(plt.MaxNLocator(3))
        ax1a.yaxis.set_major_locator(plt.MaxNLocator(3))
        ax1.grid(True)
        ax1a.grid(True)
        ax1b.grid(True)
        if k_log:
        #     y_ticks = np.logspace(-19, np.log10(k_axes_lim[1]), num=10)
        #     print(f'y_ticks {y_ticks}')
        #     ax1b.set_yticks(y_ticks)
            minor_locator = mpl.ticker.LogLocator(base=10.0, subs=np.arange(1, 11) * 0.1, numticks=100)
            ax1b.yaxis.set_minor_locator(minor_locator)
            print("Current y-axis ticks:", ax1b.get_yticks())
        #     y_ticks = np.arange(0,3.2e-16,1e-18)
        #     print(f'y_ticks {y_ticks}')
        #     ax1b.set_yticks(y_ticks)#,y_ticks)
        #     ax1b.set_yticklabels(y_ticks)
        else:
            ax1b.yaxis.set_major_locator(plt.MaxNLocator(3))
        #     ax1b.ticklabel_format(style='plain', axis='y')  # turn off the 1e-16 at the top
        #     formatter = mpl.ticker.FuncFormatter(lambda x, _: f'{x:.0f}×10⁻¹⁶')  # custom formatter
        #     ax1b.yaxis.set_major_formatter(formatter)

        ax1a.tick_params(axis='y', colors='g')
        ax1b.tick_params(axis='y', colors='dimgray') 
        
        #  hide numbers on x axes for the top two plots
        ax1.set_xticklabels([])
        ax1a.set_xticklabels([])

        if True:  # True = hide ticks, False = plot ticks
            ax1.set_yticklabels([])
            ax1.tick_params(axis='y', which='both', length=0) # remove the ticks themselves
        #     ax1.set_yticks([])
            ax1a.set_yticklabels([])
            ax1a.tick_params(axis='y', which='both', length=0) # remove the ticks themselves
        #     ax1a.set_yticks([])
            ax1b.set_yticklabels([])
        #     ax1b.set_yticks([])

        #     ax1.set_yticklabels([])
        #     ax1a.set_yticklabels([])
            
            if not k_log:
                ax1b.set_yticklabels([])
                ax1b.tick_params(axis='y', which='both', length=0) # remove the ticks themselves
        #     ax1.tick_params(axis='x', which='both', length=0) # remove the ticks themselves
        #     ax1.tick_params(axis='y', which='both', length=0) # remove the ticks themselves
        #     ax1a.tick_params(axis='y', which='both', length=0) # remove the ticks themselves
        plt.tight_layout(pad=0)

        print(f'max Pf {max(all_data_v[vars_to_plot[0]][1])}')
        print(f'max phi {max(all_data_v[vars_to_plot[1]][1])}')

        fig.tight_layout()  # must go before fill_between

        ax1.autoscale(False)


        # print(f'horizontal min {min(all_data_h[vars_to_plot[2]][1])}, max {max(all_data_h[vars_to_plot[2]][1])}, \nvertical min {min(all_data_v[vars_to_plot[2]][1])}, max {max(all_data_v[vars_to_plot[2]][1])}')

        # fig.suptitle(str(myfile))
        if k_log:
                fig_name = "phi_Pf0_separ_0.8_"+str(filenum)+"_log.png"
                transp = False
        else:
                fig_name = "phi_Pf0_separ_0.8_"+str(filenum)+".png"
                transp = True
        # plt.tight_layout()
        plt.savefig(fig_name,dpi=600,transparent=transp)
        # plt.show()
        # break

print("Done :)")
# plt.show()

