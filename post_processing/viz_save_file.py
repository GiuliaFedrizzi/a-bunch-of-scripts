"""
save a screenshot of my_experiment*.csv
"""


# I use this ^ to run python in VS code in interactive mode
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import glob
import seaborn as sns
from pathlib import Path
import re   # regex
import sys
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, to_rgb, Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.cm as cm
from matplotlib.colorbar import ColorbarBase
import colorsys


sys.path.append('/home/home01/scgf/myscripts/post_processing')   # where to look for useful_functions.py

from useful_functions import * 
# from useful_functions_moments import *

def adjust_colours(colours_from_paraview):
    adjusted_colours=[]
    for colour in colours_from_paraview:
        # Convert RGB to HSV, adjust brightness and saturation, convert back to RGB
        r, g, b = to_rgb(colour)
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        s = min(s * 1.4, 1)  # saturation
        v = min(v * 1.3, 1)  # brightness
        adjusted_rgb = colorsys.hsv_to_rgb(h, s, v)
        adjusted_colours.append(adjusted_rgb)
    return adjusted_colours

#  ---- Broken bonds colormap ---- 
colours_from_paraview_bb = [[0, 0, 0],[0.441822, 0.289241, 0.289241],[0.698582, 0.45733, 0.45733],[0.791292, 0.591516, 0.54112],[0.866968, 0.801687, 0.646762],[0.936549, 0.936549, 0.79459],[0.980196, 0.980196, 0.939336]]
adjusted_colours_bb = adjust_colours(colours_from_paraview_bb)
palette_option_bb = {i: colour for i, colour in enumerate(adjusted_colours_bb)}  # create a dictionary


#  ---- Porosity colormap ---- 
# lower_bound = 0.1988
# upper_bound = 0.4069
# intermediate = lower_bound + 0.8*(upper_bound-lower_bound) 
# # Calculate the normalized position of the intermediate point
# intermediate_normalised = (intermediate - lower_bound) / (upper_bound - lower_bound)

# Define the colormap
# colours_por = [[0.101961, 0.101961, 0.101961], [0.843368, 0.843368, 0.843368], [0.403922, 0.0, 0.121569]] # from paraview
colours_por = [[0.101961, 0.101961, 0.101961], [0.843368,0.843368,0.843368],[0.67451,0.486275,0.290196], [0.403922, 0.0, 0.121569]]
# (26,26,26) = (0.101960784,0.101960784,0.101960784)
# rgba(185,160,111,255)  = 
# rgba(26,26,26,255)  rgba(103,0,31,255)   = (0.403921569,0,0.121568627)
colours_por = adjust_colours(colours_por)
positions = [0, 0.735,0.87, 1]
colours_and_positions = list(zip(positions,colours_por))
porosity_cmap = LinearSegmentedColormap.from_list("porosity_colormap", colours_and_positions)



def read_calculate_plot(filename,palette_option_bb,porosity_cmap,poro_range):
    """
    read the csv file, plot fractures
    """
    timestep_number = int(re.findall(r'\d+',filename)[0])   # regex to find numbers in string. Then convert to float. Will be used to get "time" once multiplied by timestep.

    bb_df = pd.read_csv(filename, header=0)  # build the dataframe from the csv file
    
    # broken bonds
    sns.scatterplot(data=bb_df,x="x coord",y="y coord",hue="Broken Bonds",marker='h',s=4,palette=palette_option_bb,linewidth=0,legend=False).set_aspect('equal') # I've stopped saving "Fractures"
    
    plt.axis('off')
    plt.savefig("viz_bb_"+str(timestep_number).zfill(6)+".png",dpi=150, bbox_inches='tight', pad_inches=0)
    plt.clf()
    
    # porosity
    norm = Normalize(vmin=poro_range[0], vmax=poro_range[1])
    sm = plt.cm.ScalarMappable(cmap=porosity_cmap, norm=norm)
    sm.set_array([])
    # Calculate the colors for each point based on 'Porosity'

    # bb_df['colors'] = bb_df['Porosity'].apply(lambda x: cmap.to_rgba(x))

    sns.scatterplot(data=bb_df,x="x coord",y="y coord", hue='Porosity',
        palette=porosity_cmap,marker='h',s=4,linewidth=0,hue_norm=(poro_range[0],poro_range[1]),legend=False).set_aspect('equal') # I've stopped saving "Fractures"
    # sns.scatterplot(data=bb_df,x="x coord",y="y coord", hue='Porosity', palette=porosity_cmap,marker='h',s=4,linewidth=0,vmin=poro_range[0],vmax=poro_range[1],legend=False).set_aspect('equal') # I've stopped saving "Fractures"
    # plt.colorbar(sm, label='Porosity')
    plt.axis('off')
    
    # colormap options:
    mapper = ScalarMappable(norm=norm, cmap=porosity_cmap)
    cbar = plt.colorbar(mapper, fraction=0.05, pad=0.02)
    cbar.set_label('Porosity (Melt Fraction)', rotation=270, labelpad=15)
    # colorbar = ColorbarBase(ax, cmap=porosity_cmap, norm=norm, orientation='vertical')
    cbar.set_ticks([poro_range[0],(poro_range[0]+poro_range[1])/2,poro_range[1]])
    # cbar.set_ticks([poro_range[0],(poro_range[0]+poro_range[1])/4, (poro_range[0]+poro_range[1])/2,(poro_range[0]+poro_range[1])*3/4, poro_range[1]])
    
    plt.savefig("viz_por_"+str(timestep_number).zfill(6)+".png",dpi=150, bbox_inches='tight', pad_inches=0)
    plt.clf()
    # plt.show()

# Get the number from the filename
def extract_number(filename):
    match = re.search(r"(\d+)", filename)
    if match:
        return int(match.group(1))
    return 0

print(os.getcwd())

# default
poro_range = [0, 0]

def try_to_read_poro_file(poro_range_file):
    print(f'reading file: {poro_range_file}')
    with open(poro_range_file) as f:
        for l in f: 
            poro_range_read.append(float(l))
    return poro_range_read

poro_range_read = [] # the porosity range read from a file

# try a few paths:
if os.path.isfile('global_porosity_range.txt'):
    poro_range_file = 'global_porosity_range.txt'
    poro_range = try_to_read_poro_file(poro_range_file)
elif os.path.isfile('../global_porosity_range.txt'):
    poro_range_file = '../global_porosity_range.txt'
    poro_range = try_to_read_poro_file(poro_range_file)
elif os.path.isfile('../../global_porosity_range.txt'):
    poro_range_file = '../../global_porosity_range.txt'
    poro_range = try_to_read_poro_file(poro_range_file)
elif os.path.isfile('../../../global_porosity_range.txt'):
    poro_range_file = '../../../global_porosity_range.txt'
    poro_range = try_to_read_poro_file(poro_range_file)
else:
    poro_range = [0, 0]

# check that the range is valid
# if it's not, set it to the default
if len(poro_range) != 2 or poro_range[0] >= 1.0 or poro_range[1] >= 1.0 or poro_range[0] <= 0.0 or poro_range[1] <= 0.0:
    print(f'problem! range is {poro_range}')
    poro_range = [0, 0]

print(f'poro range: {poro_range}')

for filename in sorted(glob.glob("my_experiment*.csv"),key=extract_number, reverse=True):
    """ loop through files"""
    print(f'filename: {filename}')
    # get the porosity range
    if poro_range[0] == 0 and poro_range[1] == 0:
        df = pd.read_csv(filename, header=0)
        poro_range[0] = min(df["Porosity"])
        poro_range[1] = max(df["Porosity"])
    # print(f'range: {poro_range}')

    myfile = Path(os.getcwd()+'/'+filename)  # build file name including path
    if myfile.is_file():
        read_calculate_plot(filename,palette_option_bb,porosity_cmap,poro_range)
    

