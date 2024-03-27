"""
save a screenshot of my_experiment*.csv
"""

""" 
Save a figure based on broken bonds

made for wavedec2022, to be analysed in imagej

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
from matplotlib.colors import LinearSegmentedColormap, to_rgb
import colorsys


sys.path.append('/home/home01/scgf/myscripts/post_processing')   # where to look for useful_functions.py

from useful_functions import * 
from useful_functions_moments import *
 

# brightened_colors = [(np.array(to_rgb(color)) * 0.9 + 0.1).clip(max=1) for color in colors]  # Increase brightness
colours_from_paraview = [[0, 0, 0],[0.441822, 0.289241, 0.289241],[0.698582, 0.45733, 0.45733],[0.791292, 0.591516, 0.54112],[0.866968, 0.801687, 0.646762],[0.936549, 0.936549, 0.79459],[0.980196, 0.980196, 0.939336]]

adjusted_colours=[]
for colour in colours_from_paraview:
    # Convert RGB to HSV, adjust brightness and saturation, convert back to RGB
    r, g, b = to_rgb(colour)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    s = min(s * 1.4, 1)  # saturation
    v = min(v * 1.3, 1)  # brightness
    adjusted_rgb = colorsys.hsv_to_rgb(h, s, v)
    adjusted_colours.append(adjusted_rgb)
palette_option_bb = {i: colour for i, colour in enumerate(adjusted_colours)}  # create a dictionary
# palette_option = ['white','black']  # this one works on my laptop
# palette_option = {-1:'w',0:'b'}       # this one works on ARC. Now saves in BLUE!!
# palette_option_bb = {0:[0, 0, 0], 1:[0.441822, 0.289241, 0.289241], 2:[0.698582, 0.45733, 0.45733], 3:[0.791292, 0.591516, 0.54112], 4:[0.866968, 0.801687, 0.646762], 5:[0.936549, 0.936549, 0.79459], 6:[0.980196, 0.980196, 0.939336]}  # no broken bonds = blue,  bb >=1  = white


# Create the colormap
# custom_pink = LinearSegmentedColormap.from_list("custom_pink", colors)

# Test the colormap
# plt.figure(figsize=(8, 1))
# plt.imshow([list(range(len(colors)))], cmap=cmap, aspect="auto")
# plt.axis("off")
# plt.show()


def read_calculate_plot(filename,input_tstep,palette_option_bb):
    """
    read the csv file, plot fractures
    """
    timestep_number = int(re.findall(r'\d+',filename)[0])   # regex to find numbers in string. Then convert to float. Will be used to get "time" once multiplied by timestep.

    bb_df = pd.read_csv(filename, header=0)  # build the dataframe from the csv file

    time = str(input_tstep*timestep_number)
    print(f'time {time}, tstep: {input_tstep}')
    
    sns.scatterplot(data=bb_df,x="x coord",y="y coord",hue="Broken Bonds",marker='h',s=4,palette=palette_option_bb,linewidth=0,legend=False).set_aspect('equal') # I've stopped saving "Fractures"

    plt.title(time)
    plt.tight_layout()
    plt.savefig("viz_"+str(timestep_number).zfill(6)+".png",dpi=200)
    plt.clf()
    # plt.show()

print(os.getcwd())

input_tstep = float(getTimeStep("input.txt"))

for filename in sorted(glob.glob("my_experiment64700.csv")):
    """ loop through files"""
    print(f'filename: {filename}')

    myfile = Path(os.getcwd()+'/'+filename)  # build file name including path
    if myfile.is_file():
        read_calculate_plot(filename,input_tstep,palette_option_bb)
    

