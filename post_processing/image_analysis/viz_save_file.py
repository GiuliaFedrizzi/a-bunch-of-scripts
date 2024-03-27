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
    s = min(s * 1.1, 1)
    v = min(v * 1.1, 1)
    adjusted_rgb = colorsys.hsv_to_rgb(h, s, v)
    adjusted_colours.append(adjusted_rgb)
palette_option_bb = {i: colour for i, colour in enumerate(adjusted_colours)}
# palette_option = ['white','black']  # this one works on my laptop
# palette_option = {-1:'w',0:'b'}       # this one works on ARC. Now saves in BLUE!!
# palette_option_bb = {0:[0, 0, 0], 1:[0.441822, 0.289241, 0.289241], 2:[0.698582, 0.45733, 0.45733], 3:[0.791292, 0.591516, 0.54112], 4:[0.866968, 0.801687, 0.646762], 5:[0.936549, 0.936549, 0.79459], 6:[0.980196, 0.980196, 0.939336]}  # no broken bonds = blue,  bb >=1  = white
# import matplotlib.pyplot as plt

# Define the points in the colormap. Each point has the format:
# (normalized_position, (r, g, b))
# Note: The `x` values are adjusted from [-1, 1] to [0, 1]
# colors = [
#     (0.0, (0, 0, 0)),                                        # 0
#     # ((-0.87451 + 1) / 2, (0.312416, 0.204524, 0.204524)),  # 1
#     ((-0.74902 + 1) / 2, (0.441822, 0.289241, 0.289241)),    # 2
#     # ((-0.623529 + 1) / 2, (0.54112, 0.354246, 0.354246)),  # 3
#     ((-0.498039 + 1) / 2, (0.624831, 0.409048, 0.409048)),  # 4
#     # ((-0.372549 + 1) / 2, (0.698582, 0.45733, 0.45733)),  # 5
#     ((-0.247059 + 1) / 2, (0.764404, 0.502282, 0.500979)),  # 6
#     # ((-0.121569 + 1) / 2, (0.791292, 0.591516, 0.54112)),  # 7
#     ((0.00392157 + 1) / 2, (0.817297, 0.66895, 0.578481)),  # 8
#     # ((0.129412 + 1) / 2, (0.842499, 0.738308, 0.613572)),  # 9
#     ((0.254902 + 1) / 2, (0.866968, 0.801687, 0.646762)),  # 10
#     # ((0.380392 + 1) / 2, (0.890766, 0.86041, 0.678329)),  # 11
#     ((0.505882 + 1) / 2, (0.913944, 0.913944, 0.711254)),  # 12
#     # ((0.631373 + 1) / 2, (0.936549, 0.936549, 0.79459)),  # 13
#     ((0.756863 + 1) / 2, (0.958621, 0.958621, 0.869979)),  # 14
#     # ((0.882353 + 1) / 2, (0.980196, 0.980196, 0.939336)),  # 15
#     (1.0, (1, 1, 1)),                                        # 16
# ]

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

    # if 'Fractures' in bb_df.columns:
    #     sns.scatterplot(data=bb_df,x="x coord",y="y coord",hue="Fractures",marker='.',s=9,palette=palette_option,linewidth=0,legend=False).set_aspect('equal') #,alpha=0.8  hue="Fracrures",
    # elif 'Fracrures' in bb_df.columns:  # account for a spelling mistake: some old simulations have "Fracrures"
    #     sns.scatterplot(data=bb_df,x="x coord",y="y coord",hue="Fracrures",marker='.',s=9,palette=palette_option,linewidth=0,legend=False).set_aspect('equal') #,alpha=0.8  hue="Fracrures",
    
    sns.scatterplot(data=bb_df,x="x coord",y="y coord",hue="Broken Bonds",marker='.',s=9,palette=palette_option_bb,linewidth=0,legend=False).set_aspect('equal') # I've stopped saving "Fractures"

    plt.title(time)
    plt.tight_layout()
    plt.savefig("viz_"+str(timestep_number).zfill(6)+".png",dpi=300)
    plt.clf()
    # plt.show()

print(os.getcwd())

input_tstep = float(getTimeStep("input.txt"))

for filename in sorted(glob.glob("my_experiment44700.csv")):
    """ loop through files"""
    print(f'filename: {filename}')

    myfile = Path(os.getcwd()+'/'+filename)  # build file name including path
    if myfile.is_file():
        read_calculate_plot(filename,input_tstep,palette_option_bb)
    

