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

from viz_functions import gallery,build_array,build_melt_labels_t,set_ax_options,find_variab,setup_array,find_dirs
from useful_functions import getSaveFreq

variab = find_variab()
rose = False
im_length = 500
im_height = 506
save_freq = int(getSaveFreq())
# times = range(1000,1011,5)  # (start, end, step)

# file number
# times = list(range(1, 20, 1)) + list(range(20, 141, 5))+ list(range(150, 200, 10))  + list(range(200, 501, 20)) + list(range(500, 801, 20)) #+ list(range(850, 1500, 40))
times = list(range(24,35,1)) +  list(range(35, 110, 5)) + list(range(110, 310, 10))# + list(range(225, 420, 25))

# translate to timestep
# times *= save_freq 
times = [i*save_freq for i in times] 

x_variable = find_dirs()
# print(f'First directory: {x_variable[0]}')
os.chdir(x_variable[0])
melt_labels = find_dirs()
melt_labels.reverse()
# print(f'melt labels {melt_labels}, melt_labels[0] {melt_labels[0]}')
if '_mR_0' in melt_labels[0]:
    melt_labels = [ '0.00'+i.split('mR_0')[1] for i in melt_labels]   # from full name of directory to 0.001, 0.002 etc
else:
    melt_labels = [ '0.00'+i.split('mR0')[1] for i in melt_labels]   # from full name of directory to 0.001, 0.002 etc
# print(f'melt_labels {melt_labels}')
os.chdir('..')
# print(melt_labels)

if variab == "viscosity":
    x_variable = [i.split('_')[2] for i in x_variable]  # take the third part of the string, the one that comes after _     -> from visc_1_1e1 to 1e1
elif variab == "def_rate":
    x_variable = [i.split('def')[1] for i in x_variable]  # take the second part of the string, the one that comes after def     -> from pdef1e8 to 1e8

    
# print(x_variable)

# if variab == "viscosity":
#     # x_variable = ['1e1']  # the values of the x variable to plot (e.g. viscosity)
#     # x_variable = ['1e1','1e15','1e2','1e25','1e3','1e35','1e4']#,'2e4','4e4']  # the values of the x variable to plot (e.g. viscosity)
#     x_variable = ['1e1','1e25','1e4']#,'2e4','4e4']  # the values of the x variable to plot (e.g. viscosity)
#     #x_variable = ['1e1','5e1','1e2','5e2','1e3','5e3','1e4']#,'2e4','4e4']  # the values of the x variable to plot (e.g. viscosity)
# elif variab == "def_rate":
#     x_variable = ['1e8','5e8','9e8']#,'3e11','4e11']  # the values of the x variable to plot (e.g. def rate)
#     # x_variable = ['1e8','2e8','3e8','4e8','5e8','6e8','7e8','8e8','9e8']#,'3e11','4e11']  # the values of the x variable to plot (e.g. def rate)



ncols = len(x_variable)*2
for t in times:
    array,melt_and_time = setup_array(x_variable,melt_labels,t,im_length,im_height,variab,rose,save_freq)   # load all files
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
