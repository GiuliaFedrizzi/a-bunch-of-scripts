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
from matplotlib.ticker import FuncFormatter

# from viz_functions import gallery,build_array,build_melt_labels_t,set_ax_options,find_variab,setup_array_const_strain
from viz_functions import gallery,build_array,build_melt_labels_t,set_ax_options,find_variab,setup_array,find_dirs,setup_array_const_strain
from useful_functions import getSaveFreq,getParameterFromLatte
save_freq = int(getSaveFreq())

# variab = "def_rate"  # options: def_rate, viscosity
variab = find_variab()
rose = False
im_length = 500 # was 875
im_height = 506 # was 883

target_mr_def_ratios = [3.0,2.5,2.0,1.5,1.0,1/2,1/3,1/5,1/6,1/7]  # save figures with this melt rate - deformation rate ratio (= constant strain)
times = list(range(50, 141, 5)) + list(range(150, 501, 20)) #+ list(range(500, 801, 20)) #+ list(range(850, 1500, 40))  # (start, end, step)
times = [i*save_freq for i in times] 

melt_labels = ['0.009','0.007','0.005','0.004','0.003','0.002','0.001'] 
# melt_labels = ['0.009','0.008','0.007','0.006','0.005','0.004','0.003','0.002','0.001'] 
# melt_labels = ['0.008','0.006','0.004','0.002'] 
# x_variable = ['1e8','2e8','3e8','4e8','5e8','6e8','7e8','8e8','9e8']  # the values of the x variable to plot (e.g. def rate)
x_variable_dir_names = find_dirs()  # the name of the directories with the def rate values
x_variable = []   # initialise: will be the list of values like '1e8','2e8','3e8','4e8' etc

for x in x_variable_dir_names:
    x_value = float(getParameterFromLatte(x+'/baseFiles/input.txt','defRate'))
    if x_value != 0:
        x_variable.append(x_value)
print(f'x_variable {x_variable}')
ncols = len(x_variable)*2
for t in times:
    array = setup_array_const_strain(x_variable,melt_labels,t,target_mr_def_ratios,im_length,im_height,variab,rose,save_freq)   # load all files
    result = gallery(array,ncols)           # organise in gallery view
    fig, ax = plt.subplots()
    ax.imshow(result.astype('uint8')) # uint8 explanation: https://stackoverflow.com/questions/49643907/clipping-input-data-to-the-valid-range-for-imshow-with-rgb-data-0-1-for-floa

    set_ax_options(ax,variab,x_variable,melt_labels,t,im_length,rose)
    ax.set_yticks([(i+0.5)*im_height for i in range(0,len(target_mr_def_ratios),1)])  # 883 is height of image. Start from 0.5 times the height etc.
    formatted_labels = [f"{num:.2f}" for num in target_mr_def_ratios]
    ax.set_yticklabels(formatted_labels,fontsize=4)   #  position and values of ticks on the y axis. Start: 441 (half of image height) End: height of image times num of images in one column, Step: 883 (height of image)

    if not os.path.exists('images_in_grid'):
        os.makedirs('images_in_grid')
    if variab == "viscosity":
        filename = 'images_in_grid/cs_visc_mRate_t'
    elif variab =="def_rate":
        filename = 'images_in_grid/cs_defRate_mRate_t'
        ax.set_ylabel('Melt rate / def rate')
    plt.savefig(filename+str(int(t)).zfill(3)+'.png',dpi=600)
    
    #plt.show()

    plt.clf()   # close figure