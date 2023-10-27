'''
input: range of variables e.g. melt rate and viscosity
builds a table with images of rose diagrams
time is chosen to aways have the same amount of melt. e.g. if melt rate = 0.01, t = 50; if melt rate = 0.02, t = 25

'''
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

from viz_functions import gallery,build_array,build_melt_labels_t,set_ax_options,find_variab,setup_array_const_strain

variab = find_variab()
rose = True
im_length = 1600
im_height = 1600
times = range(10,91,10)  # (start, end, step)
times = [t*1000 for t in times]
melt_labels = ['0.009','0.008','0.007','0.006','0.005','0.004','0.003','0.002','0.001'] 
# melt_labels = ['0.008','0.006','0.004','0.002'] 
target_mr_def_ratios = [3.0,2.0,1.0,1/2,1/3]  # save figures with this melt rate - deformation rate ratio (= constant strain)

x_variable = ['1e8','2e8','3e8','4e8','5e8','6e8','7e8','8e8','9e8']#,'3e11','4e11']  # the values of the x variable to plot (e.g. def rate)


ncols = len(x_variable)#*2
for t in times:
    array,mr_def_ratio = setup_array_const_strain(x_variable,melt_labels,t,target_mr_def_ratios,im_length,im_height,variab,rose)   # load all files
    result = gallery(array,ncols)           # organise in gallery view
    fig, ax = plt.subplots()
    ax.imshow(result.astype('uint8')) # uint8 explanation: https://stackoverflow.com/questions/49643907/clipping-input-data-to-the-valid-range-for-imshow-with-rgb-data-0-1-for-floa

    set_ax_options(ax,variab,x_variable,melt_labels,t,im_length,rose)
    ax.set_yticklabels(target_mr_def_ratios,fontsize=4)   #  position and values of ticks on the y axis. Start: 441 (half of image height) End: height of image times num of images in one column, Step: 883 (height of image)
    y_ticks_positions = np.arange(im_height/2,im_height*len(melt_labels)+im_height/2,im_height)
    ax.set_yticks(y_ticks_positions)
    if not os.path.exists('images_in_grid'):
        os.makedirs('images_in_grid')
    if variab == "viscosity":
        ax.set_ylabel('Melt increment per spot')
        filename = 'images_in_grid/rose_cs_visc_mRate_t'
    elif variab =="def_rate":
        filename = 'images_in_grid/rose_cs_defRate_mRate_t'
        ax.set_ylabel('Melt rate / def rate')

    plt.savefig(filename+str(int(t/1000)).zfill(3)+'.png',dpi=600)  # back to the default name that matches viz_images_in_grid
    
    #plt.show()

    plt.clf()   # close figure