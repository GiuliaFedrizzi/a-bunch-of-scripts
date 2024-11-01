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

from viz_functions import gallery,build_array,set_ax_options,find_variab,setup_array,find_dirs,setup_array_layers
from useful_functions import getSaveFreq

rose = False
im_length = 500
im_height = 506
save_freq = int(getSaveFreq())
# times = range(1000,1011,5)  # (start, end, step)

# file number# times = list(range(24,35,1)) +  list(range(35, 110, 5)) + list(range(110, 310, 10))# + list(range(225, 420, 25))
times = list(range(25,75,5))
# translate to timestep
# times *= save_freq 
times = [i*save_freq for i in times] 

viscosity_dirs = ["visc_2_1e2","visc_3_1e3"]
mr_contrast_dirs = ['lr41','lr44','lr42','lr45','lr43']
mr_contrast_dirs.reverse()
def_rate_dirs = ['def0e-8','def8e-8','def1e-7']
melt_labels = '0.005'

viscosity = ['1e1','1e15']


ncols = len(def_rate_dirs)*2
for t in times:
    array,def_rate_and_time = setup_array_layers(def_rate_dirs,mr_contrast_dirs,t,im_length,im_height,rose,save_freq)   # load all files
    result = gallery(array,ncols)           # organise in gallery view
    fig, ax = plt.subplots()
    ax.imshow(result.astype('uint8')) # uint8 explanation: https://stackoverflow.com/questions/49643907/clipping-input-data-to-the-valid-range-for-imshow-with-rgb-data-0-1-for-floa

    print(f'len x ticks {len(np.arange(im_length/2,im_length*len(def_rate_dirs)*2,im_length*2))}')
    print(f'def_rate_and_time {def_rate_and_time}')
    ax.set_xticks(np.arange(im_length/2,im_length*len(def_rate_dirs)*2,im_length*2)) #  position and values of ticks on the x axis. Start: 437 (half width of an image) End: length of image times num of images in one row, Step: 883 (legth of image)
    ax.set_xticklabels(def_rate_and_time,fontsize=10)

    y_variable = [str(i) for i in mr_contrast_dirs]   # convert to list of strings
    ax.set_yticklabels(y_variable,fontsize=10)   #  position and values of ticks on the y axis. Start: 441 (half of image height) End: height of image times num of images in one column, Step: 883 (height of image)
    y_ticks_positions = np.arange(im_height/2,im_height*len(melt_labels)+im_height/2,im_height)
    ax.set_yticks(y_ticks_positions)
    if not os.path.exists('images_in_grid_layers'):
        os.makedirs('images_in_grid_layers')

    ax.set_ylabel('Melt Rate Contrast')
    filename = 'images_in_grid_layers/layer_grid_v1e3_t'

    plt.savefig(filename+str(t).zfill(3)+'.png',dpi=600)
    
    # plt.show()

    plt.clf()   # close figure
