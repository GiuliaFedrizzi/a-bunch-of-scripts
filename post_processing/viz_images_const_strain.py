'''
input: range of variables e.g. melt rate and viscosity
builds a table with images of porosity and broken bonds
time is chosen to aways have the same amount of melt. e.g. if melt rate = 0.01, t = 50; if melt rate = 0.02, t = 25
combinations of melt rate and def rate are chosen so that they have constant strain
images are trimmed to show only the domain
'''
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

from viz_images_in_grid import *


variab = "def_rate"  # options: def_rate, viscosity

times = range(30,31,1)  # (start, end, step)
melt_labels = ['0.009','0.008','0.007','0.006','0.005','0.004','0.003','0.002','0.001'] 
x_variable = ['1e8','2e8','3e8','4e8','5e8','6e8','7e8','8e8','9e8']#,'3e11','4e11']  # the values of the x variable to plot (e.g. def rate)

