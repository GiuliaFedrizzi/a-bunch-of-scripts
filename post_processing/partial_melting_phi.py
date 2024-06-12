"""
Visualise the effect of partial melting on the porosity:
the area of the particles gets smaller
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from useful_functions import extract_two_profiles

plt.rcParams["xtick.top"] = True
plt.rcParams["xtick.labeltop"] = True
plt.rcParams["xtick.bottom"] = False
plt.rcParams["xtick.labelbottom"] = False
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams["font.weight"] = "bold"  # set default weight to bold
plt.rcParams['lines.linewidth'] = 5
plt.rcParams['axes.linewidth'] = 4

filename = 'my_experiment00000.csv'

poro_colour = ('g')
press_colour = ('#1f77b4')  # default blue  

def read_calculate_plot(filename):
    """
    read the csv file, plot
    """
    myExp_all = pd.read_csv(filename, header=0)
    # xlimits = [0.445,0.565]
    xlimits = [0.445,0.56]
    mask = myExp_all['y coord'] < xlimits[1]   # create a mask: use these values to identify which rows to keep
    myExp_all = myExp_all[mask].copy()  # keep only the rows found by mask

    mask = myExp_all['y coord'] > xlimits[0]     # create a mask: use these values to identify which rows to keep
    myExp_all = myExp_all[mask].copy()  # keep only the rows found by mask

    mask = myExp_all['x coord'] < xlimits[1]    # create a mask: use these values to identify which rows to keep
    myExp_all = myExp_all[mask].copy()  # keep only the rows found by mask

    mask = myExp_all['x coord'] > xlimits[0]    # create a mask: use these values to identify which rows to keep
    myExp = myExp_all[mask].copy()  # keep only the rows found by mask

    if True:
        myExp['solid_frac'] = np.log(1-myExp['Porosity'])
        print(max(myExp['solid_frac']),min(myExp['solid_frac']))
        plt.figure(figsize=(7,7))
        # vmin=0, vmax=6
        sns.scatterplot(data=myExp,x="x coord",y="y coord",hue="solid_frac",color='silver',linewidth=1.5,edgecolor='k',alpha=0.8,
            size="solid_frac",sizes=(20, 220)).set_aspect('equal') #legend='full'
        plt.xlabel('')
        plt.ylabel('')
        plt.legend([],[], frameon=False)
        # plt.show()
        plt.savefig("partial_melt_phi_circles.png")
        # print("done")

    if True:
        # profile
        fig, ax1 = plt.subplots(nrows=1, ncols=1,figsize=(12,8))
        ax2 = ax1.twinx()
        res = 200; res_y = 230
        point_indexes = [(res*res_y/2)-1,5]   # top (horizontal) and left (vertical). X original: res*res_y/2+res, central: res*res_y/2 (?)
        print(f'point_indexes {point_indexes}')
        
        all_data_h = {}

        for v in ["Pressure","Porosity"]:
            (x_v, y_v), (x_h, y_h) = extract_two_profiles(filename, v,point_indexes,res)
            all_data_h[v] = (x_h, y_h)
        ax1.plot(all_data_h["Pressure"][0],all_data_h["Pressure"][1],color=press_colour)  # pressure
        ax2.plot(all_data_h["Porosity"][0],all_data_h["Porosity"][1],color=poro_colour)  # porosity
        ax1.axhline(y=min(all_data_h["Pressure"][1]),color=press_colour,linestyle='--',alpha=0.6)
        ax2.axhline(y=min(all_data_h["Porosity"][1]),color=poro_colour,linestyle='--',alpha=0.6)
        solid_frac = [1-x for x in y_h]
        # plt.scatter(x_h,solid_frac)
        ax1.yaxis.set_major_locator(plt.MaxNLocator(4))  # reduce the number of annotations on the y axis
        ax2.yaxis.set_major_locator(plt.MaxNLocator(4))  # reduce the number of annotations on the y axis
        ax1.tick_params(axis='y', colors=press_colour)
        ax2.tick_params(axis='y', colors=poro_colour)  
        # ax1.tick_params(axis='x', labelsize=15)
        # ax1.set_ylabel("Fluid Pressure", color=press_colour)
        # ax2.set_ylabel("Porosity", color=poro_colour)
        plt.tight_layout()
         
        plt.xlim(xlimits)
        ax1.set_ylim([7.605e7,7.609e7])
        ax2.set_ylim([0.134,0.1343])
        ax1.spines['left'].set_color(press_colour)
        ax2.spines['left'].set_color(press_colour)
        ax2.spines['right'].set_color(poro_colour)
        plt.tight_layout()
        plt.savefig("partial_melt_phi_lines.png")
    plt.show()


myfile = Path(os.getcwd()+'/'+filename)  # build file name including path
if myfile.is_file():
    variable_hue = read_calculate_plot(filename)