'''
input: range of variables e.g. melt rate and viscosity
builds a table with images of porosity and broken bonds
time is chosen to aways have the same amount of melt. e.g. if melt rate = 0.01, t = 50; if melt rate = 0.02, t = 25
images are trimmed to show only the domain
'''
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

from viz_functions import gallery,build_array,build_melt_labels_t,set_ax_options,find_variab,setup_array

variab = find_variab()
rose = False
im_length = 875
im_height = 883
times = range(5,60,5)  # (start, end, step)
melt_labels = ['0.009','0.008','0.007','0.006','0.005','0.004','0.003','0.002','0.001'] 
# melt_labels = ['0.008','0.006','0.004','0.002'] 

if variab == "viscosity":
    x_variable = ['1e1']  # the values of the x variable to plot (e.g. viscosity)
    # x_variable = ['1e1','1e15','1e2','1e25','1e3','1e35','1e4']#,'2e4','4e4']  # the values of the x variable to plot (e.g. viscosity)
    #x_variable = ['1e1','5e1','1e2','5e2','1e3','5e3','1e4']#,'2e4','4e4']  # the values of the x variable to plot (e.g. viscosity)
elif variab == "def_rate":
    x_variable = ['1e8','2e8','3e8','4e8','5e8','6e8','7e8','8e8','9e8']#,'3e11','4e11']  # the values of the x variable to plot (e.g. def rate)



ncols = len(x_variable)*2
for t in times:
    array,melt_and_time = setup_array(x_variable,melt_labels,t,im_length,im_height,variab,rose)   # load all files
    result = gallery(array,ncols)           # organise in gallery view
    fig, ax = plt.subplots()
    ax.imshow(result.astype('uint8')) # uint8 explanation: https://stackoverflow.com/questions/49643907/clipping-input-data-to-the-valid-range-for-imshow-with-rgb-data-0-1-for-floa

    set_ax_options(ax,variab,x_variable,melt_labels,t,im_length,rose)
    ax.set_yticklabels(melt_and_time,fontsize=4)   #  position and values of ticks on the y axis. Start: 441 (half of image height) End: height of image times num of images in one column, Step: 883 (height of image)
    y_ticks_positions = np.arange(im_height/2,im_height*len(melt_labels)+im_height/2,im_height)
    ax.set_yticks(y_ticks_positions)
    if not os.path.exists('images_in_grid'):
        os.makedirs('images_in_grid')
    if variab == "viscosity":
        ax.set_ylabel('Melt increment per spot')
        filename = 'images_in_grid/visc_mRate_160_t'
    elif variab =="def_rate":
        filename = 'images_in_grid/defRate_mRate_t'
        ax.set_ylabel('Pressure increment')

    plt.savefig(filename+str(t).zfill(3)+'.png',dpi=600)
    
    #plt.show()

    plt.clf()   # close figure
