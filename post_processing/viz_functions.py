"""
Functions called by viz_images* and viz_rose* scripts.
"""

import matplotlib.pyplot as plt
import os
import math
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

def build_array(big_array,variab,x,x_val,melt_rate,file_number,row,cols,rose):
    """
    given all the properties (x variable, melt rate, time, coords in the big array),
    populate the big array that contains all the data and makes the final image
    """
    if variab == "viscosity":
        exp = x_val.split('e')[-1] # the exponent after visc_ and before 5e3 or 1e4 etc
        if len(exp)==2:
           exp = str((float(exp)/10)) # true number
        #    print(f'exp is {exp}')
        ###### path to the files
        ## 2 levels, 2 options for viscosity
        potential_file_path = 'visc_'+exp+'_'+x_val+'/vis1e2_mR_'+melt_rate#+'/'
        if os.path.isdir(potential_file_path) == False:
            potential_file_path = 'visc_'+exp+'_'+x_val+'/vis1e2_mR'+melt_rate#+'/'
            if os.path.isdir(potential_file_path) == False:
                potential_file_path ='visc_'+exp+'_'+x_val+'/vis'+x_val+'_mR_'+melt_rate#+'/'
                # if os.path.isdir(potential_file_path) == False:
                    #print("I've tried twice without success")
                if os.path.isdir(potential_file_path) == False:
                    potential_file_path ='visc_'+exp[0]+'_'+x_val+'/vis'+x_val+'_mR_'+melt_rate#+'/'
                    if os.path.isdir(potential_file_path) == False:
                        potential_file_path ='visc_'+exp[0]+'_'+x_val+'/vis5e'+exp[0]+'_mR_'+melt_rate#+'/'  # e.g. vis5e2_mR_01

        # poro_file = 'wd05_visc/visc_'+exp+'_'+x_val+'/vis'+x_val+'_mR_0'+str(melt_rate)+'/a_porosity_'+file_number+'.png' 
        # bb_file = 'wd05_visc/visc_'+exp+'_'+x_val+'/vis'+x_val+'_mR_0'+str(melt_rate)+'/a_brokenBonds_'+file_number+'.png'
        # print(f'path: {potential_file_path}')
        ##  only 1 level of subdirs
        # poro_file = 'vis'+x_val+'_mR_'+melt_rate+'/a_porosity_'+file_number+'.png' 
        # bb_file = 'vis'+x_val+'_mR_'+melt_rate+'/a_brokenBonds_'+file_number+'.png'
    elif variab == "def_rate":
        potential_file_path = 'thdef'+x_val+'/vis1e2_mR_'+ melt_rate   # transfer zone
        if os.path.isdir(potential_file_path) == False:
            potential_file_path ='pdef'+x_val+'/vis1e2_mR_'+ melt_rate#+'/'  # production zone
            if os.path.isdir(potential_file_path) == False:
                potential_file_path ='pdef'+x_val+'/vis1e2_mR'+ melt_rate#+'/'  # production zone
    # now we have the path, choose if I open "poro_file" and "bb_file" or "rose_file"
    if rose == False:
        """ "poro_file" and "bb_file"  """
        poro_file = potential_file_path +'/a_porosity_160_'+file_number+'.png' 
        # bb_file  = potential_file_path +'/a_brokenBonds_'+file_number+'.png'
        bb_file  = potential_file_path +'/viz_bb_'+file_number+'.png'
    
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

    else:
        """ here "rose_file" only """
        rose_file = potential_file_path +'/rose_weight_p_top_py_bb_'+file_number+'_nx.png' # try "top" first    
        if os.path.isfile(rose_file) == False:
            """ try another version """
            rose_file = potential_file_path +'/rose_weight_p_py_bb_'+file_number+'_nx.png'   # e.g.  rose_norm_p_py_bb_026000_nx
        if os.path.isfile(rose_file):
            print(f'found file {rose_file}')
            rose_big_file = Image.open(rose_file)  # to do: do I need to crop it?

            w, h = rose_big_file.size
            # print(f'w = {w}, h = {h}')
            rf  =  np.asarray(rose_big_file.convert('RGB'))
            # rf  =  np.asarray(bb_big_file.crop((585,188,1468,1063)).convert('RGB'))
            big_array[row*cols+x,:,:,:] = rf  # only 1 figure per simulation now
            # print(row*cols+2*x+1)
            rose_big_file.close()
        else:
            print("no file called ",rose_file)
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


def set_ax_options(ax,variab,x_variable,melt_labels,t,im_length,rose):
    """
    set ticks, labels, title etc for the final plot
    """
    if rose:
        ax.set_xticks(np.arange(im_length/2,im_length*len(x_variable),im_length)) #  position and values of ticks on the x axis. Start: 437 (half width of an image) End: length of image times num of images in one row, Step: 883 (legth of image)
    else:
        ax.set_xticks(np.arange(im_length/2,im_length*len(x_variable)*2,im_length*2)) #  position and values of ticks on the x axis. Start: 437 (half width of an image) End: length of image times num of images in one row, Step: 883 (legth of image)

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

def find_variab():
    if True in ['def' in x for x in os.listdir('.')]:  # if at least one of the directories contains "def",
        variab = "def_rate"    # it means that the variable is deformation rate
    elif True in ['visc' in x for x in os.listdir('.')]:
        variab = "viscosity"  # in this case, variable is viscosity
    else:
        raise ValueError('Could not find directories with names that tells me what the variable is')
    return variab

def setup_array(x_variable,melt_labels,t,im_length,im_height,variab,rose):
    """ setup the big array with dimensions 
         - rows*cols, 
         - length and 
         - height of each image, 
         - 3 (RGB) 

         calls build_array(), which populates the big array
         """
    print(melt_labels)
    melt_labels_t = [None] * len(melt_labels) # labels with extra term for time. Empty string list, same size as melt_labels
    rows = len(melt_labels)  # how many values of melt rate
    if rose==False:
        cols = len(x_variable)*2
    else:
        cols = len(x_variable)
    # initialise the big array
    big_array = np.full(shape=(rows*cols, im_length, im_height, 3),fill_value=255)  # 1st number is the number of images to display, then size, then colour channels (RGB). Set initial values to 255 (white)

    # for row in range(0,rows):
    for row,melt_rate in enumerate(melt_labels):
        melt_rate = melt_rate.replace("0.0","")   # e.g. from 0.001 to 01, or from 0.02 to 2
        file_number = str(round(t/(int(melt_rate)))).zfill(5)  # normalise t by the melt rate.  .zfill(5) fills the string with 0 until it's 5 characters long
        if rose:
            file_number = str(round(int(file_number)/1000)*1000).zfill(6)
        for x,x_val in enumerate(x_variable):
            big_array = build_array(big_array,variab,x,x_val,melt_rate,file_number,row,cols,rose)   # fill the array with data from images!
            # ----------------------------------
        melt_labels_t[row] = build_melt_labels_t(melt_labels,t,file_number,row)
        print(melt_labels_t[row])

    return big_array,melt_labels_t  # the number of columns is the num of x variable times two (two images for each sim)
    # return np.array([bb,poro,bb,poro])


def setup_array_const_strain(x_variable,melt_labels,t,target_mr_def_ratios,im_length,im_height,variab,rose):
    """ setup the 'big array' with dimensions:
         - rows*cols, 
         - length and 
         - height of each image, 
         - 3 (RGB) 
        Keeps strain constant (not just melt amount)

         calls build_array(), which populates the big array """
    print(melt_labels)
    melt_labels_t = [None] * len(melt_labels) # labels with extra term for time. Empty string list, same size as melt_labels
    rows = len(target_mr_def_ratios)  # how many values of ratios: it's the number of rows
    if rose==False:
        cols = len(x_variable)*2
    else:
        cols = len(x_variable)

    # initialise the big array
    big_array = np.full(shape=(rows*cols, im_length, im_height, 3),fill_value=255)  # 1st number is the number of images to display, then size, then colour channels (RGB). Set initial values to 255 (white)

    for melt_rate in melt_labels:
        melt_rate = melt_rate.replace("0.0","")   # e.g. from 0.001 to 01, or from 0.02 to 2
        norm_time = t/(float(melt_rate))
        # print(f'norm time: {norm_time}')
        file_number = str(round(norm_time)).zfill(5)  # normalise t by the melt rate.  .zfill(5) fills the string with 0 until it's 5 characters long
        if rose:
            file_number = str(round(int(file_number)/1000)*1000).zfill(6)
        # print(f'file n {file_number}')
        # if float(melt_rate)/

        for x,x_val in enumerate(x_variable):
            real_mr_def_ratio = float(melt_rate)/(float(x_val)*1e-8)  # def rate is still in the 1e08 form (exp is actually negative)
            # print(f'melt_rate: {melt_rate}, def rate: {x_val}, ratio {real_mr_def_ratio}')
            for n,current_mr_def_ratio in enumerate(target_mr_def_ratios):
                # to have the same strain, we need the melt rate/defRate ratio to be fixed
                # n is the row number, one per target melt ratio

                if math.isclose(real_mr_def_ratio, current_mr_def_ratio, rel_tol=0.1):
                    print(f'-  real: {real_mr_def_ratio}, current ideal: {current_mr_def_ratio}, diff {real_mr_def_ratio-current_mr_def_ratio}')
                    big_array = build_array(big_array,variab,x,x_val,melt_rate,file_number,n,cols,rose)   # fill the array with data from images!
                # otherwise skip it
                # ----------------------------------
        print("mrate ",melt_rate,"\n")
        # end of viscosity/deformation loop


    return big_array  # the number of columns is the num of x variable times two (two images for each sim)
    

def find_dirs(dir_type):
    directories_list = []
    # if dir_type == "def_rate":
    # List to hold the names of directories

    # Iterate over the items in the current directory
    for item in os.listdir('.'):
        # Check if the item is a directory and not one of the excluded names
        if os.path.isdir(item) and item not in ["baseFiles", "images_in_grid", "branch_plots"]:
            directories_list.append(item)
    directories_list.sort()
    # Printing the list (optional)
    print("Directories found:", directories_list)
    return directories_list