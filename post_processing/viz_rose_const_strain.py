'''
input: range of variables e.g. melt rate and viscosity
builds a table with images of rose diagrams
time is chosen to aways have the same amount of melt. e.g. if melt rate = 0.01, t = 50; if melt rate = 0.02, t = 25

'''
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

from viz_functions import gallery,build_array,build_melt_labels_t,set_ax_options,find_variab,setup_array_const_strain_dbscan,find_dirs,setup_array_const_strain
from useful_functions import getSaveFreq,getParameterFromLatte

save_freq = int(getSaveFreq())

variab = find_variab()
rose = True
im_length = 1600
im_height = 1600

# def rate directories
x_variable_dir_names = find_dirs()  # the name of the directories with the def rate values
x_variable = []   # initialise: will be the list of values like '1e8','2e8','3e8','4e8' etc

# melt rate directories
os.chdir(x_variable_dir_names[0])
melt_labels = find_dirs()
melt_labels = [i.replace('vis1e2_mR0','0.00') for i in melt_labels]  # go from 'vis1e2_mR01' to '0.001'
melt_labels.reverse()
os.chdir('..')

print(f'melt_labels {melt_labels}')
times = list(range(200, 201, 10))# + list(range(200, 301, 20)) #list(range(50, 141, 5))   + list(range(500, 801, 20)) #+ list(range(850, 1500, 40))  # (start, end, step)
times = [i*save_freq for i in times] 

# target_mr_def_ratios = [3.0,2.5,2.0,1.5,1.0,1/2,1/3,1/4,1/5,1/6,1/7]  # save figures with this melt rate - deformation rate ratio (= constant strain)
target_mr_def_ratios = [2.0,1.5,1.0,1/2,1/3,1/4,1/5]  # values for figure in chapter 6

# for x in x_variable_dir_names:
#     x_value = float(getParameterFromLatte(x+'/baseFiles/input.txt','defRate'))
#     if x_value != 0:
#         x_variable.append(x_value)
x_variable = [2e-08, 4e-08, 6e-08, 8e-08]
print(f'x_variable {x_variable}')

ncols = len(x_variable)#*2
for t in times:
    # array = setup_array_const_strain_dbscan(x_variable,melt_labels,t,target_mr_def_ratios,im_length,im_height,variab,rose)   # load all files
    array = setup_array_const_strain(x_variable,melt_labels,t,target_mr_def_ratios,im_length,im_height,variab,rose,save_freq)   # load all files
                                           # x_variable,melt_labels,t,target_mr_def_ratios,im_length,im_height,variab,rose,save_freq 
    result = gallery(array,ncols)           # organise in gallery view
    fig, ax = plt.subplots()
    ax.imshow(result.astype('uint8')) # uint8 explanation: https://stackoverflow.com/questions/49643907/clipping-input-data-to-the-valid-range-for-imshow-with-rgb-data-0-1-for-floa

    fig.tight_layout()
    set_ax_options(ax,variab,x_variable,melt_labels,t,im_length,rose)
    formatted_labels = [f"{num:.2f}" for num in target_mr_def_ratios]
    ax.set_yticklabels(formatted_labels,fontsize=4)   #  position and values of ticks on the y axis. Start: 441 (half of image height) End: height of image times num of images in one column, Step: 883 (height of image)
    y_ticks_positions = np.arange(im_height/2,im_height*len(target_mr_def_ratios)+im_height/2,im_height)
    ax.set_yticks(y_ticks_positions)

    if not os.path.exists('images_in_grid'):
        os.makedirs('images_in_grid')
    if variab == "viscosity":
        ax.set_ylabel('Melt increment per spot')
        filename = 'images_in_grid/rose_cs_visc_mRate_t'
    elif variab =="def_rate":
        filename = 'images_in_grid/rose_cs_defRate_mRate_t'
        ax.set_ylabel('Melt rate / def rate')

    plt.savefig(filename+str(int(t/1000)).zfill(3)+'.png',dpi=2400)  # back to the default name that matches viz_images_in_grid
    
    # plt.show()

    plt.clf()   # close figure