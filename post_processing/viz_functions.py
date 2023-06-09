"""
Functions called by viz_images* scripts.
"""

import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np



def gallery(array, ncols):
    """
    set the shape of the "result" array. Returns a 'gallery' structure
    """
    nindex, height, width, intensity = array.shape
    nrows = nindex//ncols
    assert nindex == nrows*ncols
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (array.reshape(nrows, ncols, height, width, intensity)
              .swapaxes(1,2)
              .reshape(height*nrows, width*ncols, intensity))
    return result

def build_array(big_array,variab,x,x_val,melt_rate,file_number,row,cols):
    """
    given all the properties (x variable, melt rate, time, coords in the big array),
    populate the big array that contains all the data and makes the final image
    """
    if variab == "viscosity":
        exp = x_val.split('e')[-1] # the exponent after visc_ and before 5e3 or 1e4 etc
        # poro_file = 'wd05_visc/visc_'+exp+'_'+x_val+'/vis'+x_val+'_mR_0'+str(melt_rate)+'/a_porosity_'+file_number+'.png' 
        # bb_file = 'wd05_visc/visc_'+exp+'_'+x_val+'/vis'+x_val+'_mR_0'+str(melt_rate)+'/a_brokenBonds_'+file_number+'.png'
        
        ## 2 levels of subdirs
        # poro_file = 'visc_'+exp+'_'+x_val+'/vis'+x_val+'_mR_'+melt_rate+'/a_porosity_'+file_number+'.png' 
        # bb_file = 'visc_'+exp+'_'+x_val+'/vis'+x_val+'_mR_'+melt_rate+'/a_brokenBonds_'+file_number+'.png'

        ## 2 levels, but "visc" value is fixed
        poro_file = 'visc_'+exp+'_'+x_val+'/vis1e2_mR_'+melt_rate+'/a_porosity_'+file_number+'.png' 
        bb_file = 'visc_'+exp+'_'+x_val+'/vis1e2_mR_'+melt_rate+'/a_brokenBonds_'+file_number+'.png'
        
        ##  only 1 level of subdirs
        # poro_file = 'vis'+x_val+'_mR_'+melt_rate+'/a_porosity_'+file_number+'.png' 
        # bb_file = 'vis'+x_val+'_mR_'+melt_rate+'/a_brokenBonds_'+file_number+'.png'
    elif variab == "def_rate":
        poro_file = 'thdef'+x_val+'/vis1e2_mR_'+melt_rate+'/a_porosity_'+file_number+'.png' # .zfill(5) fills the string with 0 until it's 5 characters long
        bb_file  = 'thdef'+x_val+'/vis1e2_mR_'+melt_rate+'/a_brokenBonds_'+file_number+'.png'

    if os.path.isfile(poro_file):
        poro_big_file = Image.open(poro_file)  # I open it here so then I can call poro_big_file.close() and close it
        poro = np.asarray(poro_big_file.crop((585,188,1468,1063)).convert('RGB'))
        big_array[row*cols+2*x,:,:,:] = poro  # 0 2 4 rows*2 because there are 2 images for each simulation. First index is for each image
        #print(row*cols+2*x)
        poro_big_file.close()
    else:
        print("no file called ",poro_file)

    if os.path.isfile(bb_file):
        bb_big_file = Image.open(bb_file)
        bb  =  np.asarray(bb_big_file.crop((585,188,1468,1063)).convert('RGB'))
        big_array[row*cols+2*x+1,:,:,:] = bb  # 1 3 5 same as above, but +1 (the next one)
        # print(row*cols+2*x+1)
        bb_big_file.close()
    else:
        print("no file called ",bb_file)
    return big_array

def build_melt_labels_t(melt_labels,t,file_number,row):
    melt_rate = melt_labels[row].replace("0.0","")   # e.g. from 0.001 to 01, or from 0.02 to 2
    exact = 0   # whether file corresponds to the specified time exactly 
    if t%(int(melt_rate)) == 0:
        exact = 1
    if exact:
        return melt_labels[row]+" ("+file_number+")"
    else:
        return melt_labels[row]+" ("+file_number+"*)"   # add a flag that says it's not exact


def set_ax_options(ax,variab,x_variable,melt_labels,t):
    """
    set ticks, labels, title etc for the final plot
    """
    ax.set_xticks(np.arange(437,875*len(x_variable)*2,875*2)) #  position and values of ticks on the x axis. Start: 437 (half width of an image) End: length of image times num of images in one row, Step: 883 (legth of image)
    ax.set_xticklabels(x_variable,fontsize=4)

    if variab == "viscosity":
        ax.set_xlabel('Viscosity')
    elif variab == "def_rate":
        ax.set_xlabel('Deformation rate')
        xlabels = [l.replace("e8","e-8") for l in x_variable]
        ax.set_xticklabels(xlabels,fontsize=4)  # overwrite labels with "-8" (true exponent) instead of 8 

    # plt.yticks(np.arange(441,883*4,883), ['0.01','0.02','0.03','0.04'])
    ax.set_title("$t_{ref}$ = "+str(t),fontsize=8)
    plt.tight_layout()
