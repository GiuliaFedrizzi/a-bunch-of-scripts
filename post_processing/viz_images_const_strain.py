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

from viz_functions import gallery,build_array,build_melt_labels_t,set_ax_options

variab = "def_rate"  # options: def_rate, viscosity
target_mr_def_ratio = 1.0  # save figures with this melt rate - deformation rate ratio (= constant strain)
times = range(30,31,1)  # (start, end, step)
melt_labels = ['0.009','0.008','0.007','0.006','0.005','0.004','0.003','0.002','0.001'] 
x_variable = ['1e8','2e8','3e8','4e8','5e8','6e8','7e8','8e8','9e8']  # the values of the x variable to plot (e.g. def rate)

def setup_array_const_strain(x_variable,melt_labels,t,target_mr_def_ratio):
    print(melt_labels)
    melt_labels_t = [None] * len(melt_labels) # labels with extra term for time. Empty string list, same size as melt_labels
    # rows = len(melt_labels)  # how many values of melt rate
    rows = 1  # how many values of melt rate
    cols = len(x_variable)*2

    # initialise the big array
    big_array = np.full(shape=(rows*cols, 875, 883, 3),fill_value=255)  # 1st number is the number of images to display, then size, then colour channels (RGB). Set initial values to 255 (white)

    # for row in range(0,rows):
    for row,melt_rate in enumerate(melt_labels):
        # print(f'rows: row = {row}')
        melt_rate = melt_rate.replace("0.0","")   # e.g. from 0.001 to 01, or from 0.02 to 2
        # melt_rate = "0"+str(row+1)
        # melt_rate = melt_labels[row].replace("0.0","")   # e.g. from 0.001 to 01, or from 0.02 to 2
        norm_time = t/(float(melt_rate))
        # print(norm_time)
        file_number = str(round(norm_time)).zfill(5)  # normalise t by the melt rate.  .zfill(5) fills the string with 0 until it's 5 characters long
        # if float(melt_rate)/

        for x,x_val in enumerate(x_variable):
            # to have the same strain, we need the melt rate/defRate ratio to be fixed
            mr_def_ratio = float(melt_rate)/(float(x_val)*1e-8)  # def rate is still in the 1e08 form (exp is actually negative)
            print(f'melt_rate: {melt_rate}, def rate: {x_val}, ratio {mr_def_ratio}')
            if mr_def_ratio == target_mr_def_ratio:
                # print("right melt - def rate ratio! I'm saving this one...")    
                # 0 is the row number (there is only 1)
                big_array = build_array(big_array,variab,x,x_val,melt_rate,file_number,0,cols)   # fill the array with data from images!
            # ----------------------------------
        print("mrate ",melt_rate)
        # end of viscosity/deformation loop


    return big_array,mr_def_ratio  # the number of columns is the num of x variable times two (two images for each sim)
    

ncols = len(x_variable)*2
for t in times:
    array,mr_def_ratio = setup_array_const_strain(x_variable,melt_labels,t,target_mr_def_ratio)   # load all files
    result = gallery(array,ncols)           # organise in gallery view
    fig, ax = plt.subplots()
    ax.imshow(result.astype('uint8')) # uint8 explanation: https://stackoverflow.com/questions/49643907/clipping-input-data-to-the-valid-range-for-imshow-with-rgb-data-0-1-for-floa

    set_ax_options(ax,variab,x_variable,melt_labels,t)
    ax.set_yticks([441])
    ax.set_yticklabels([target_mr_def_ratio],fontsize=4)   #  position and values of ticks on the y axis. Start: 441 (half of image height) End: height of image times num of images in one column, Step: 883 (height of image)
    if not os.path.exists('images_in_grid'):
        os.makedirs('images_in_grid')
    if variab == "viscosity":
        filename = 'images_in_grid/cs_visc_mRate_t'
    elif variab =="def_rate":
        filename = 'images_in_grid/cs_defRate_mRate_'+str(target_mr_def_ratio)+'t'
        ax.set_ylabel('Melt rate / def rate')
    print(filename)
    #plt.savefig(filename+str(t).zfill(3)+'.png',dpi=600)
    
    plt.show()

    plt.clf()   # close figure