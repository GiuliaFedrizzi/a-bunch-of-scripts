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

viscosity = 1
def_rate = 0

os.chdir('/nobackup/scgf/myExperiments/threeAreas/through/th47')

times = range(1,151,1)  # (start, end, step)
# melt_labels = ['0.001','0.002','0.003','0.004','0.005','0.006','0.007','0.008','0.009']  
melt_labels = ['0.001','0.003','0.005','0.007','0.009'] 



if viscosity:
    x_variable = ['1e1','5e1','1e2','5e2','1e3','5e3','1e4']#,'2e4','4e4']  # the values of the x variable to plot (e.g. viscosity)
    # x_variable = ['1e2']#,'2e4','4e4']  # the values of the x variable to plot (e.g. viscosity)
elif def_rate:
    x_variable = ['1e8','2e8','3e8','4e8','5e8','6e8','7e8','8e8','9e8']#,'3e11','4e11']  # the values of the x variable to plot (e.g. def rate)

def gallery(array, ncols):
    nindex, height, width, intensity = array.shape
    nrows = nindex//ncols
    assert nindex == nrows*ncols
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (array.reshape(nrows, ncols, height, width, intensity)
              .swapaxes(1,2)
              .reshape(height*nrows, width*ncols, intensity))
    return result

def make_array(x_variable,melt_labels,t):
    print(melt_labels)
    melt_labels_t = [None] * len(melt_labels) # labels with extra term for time. Empty string list, same size as melt_labels
    rows = len(melt_labels)  # how many values of melt rate
    cols = len(x_variable)*2

    # initialise the big array
    # big_array = np.zeros(shape=(rows*cols, 875, 883, 3))  # 1st number is the number of images to display, then size, then colour channels (RGB)
    big_array = np.full(shape=(rows*cols, 875, 883, 3),fill_value=255)  # 1st number is the number of images to display, then size, then colour channels (RGB). Set initial values to 255 (white)

    for row in range(0,rows):
        # melt_rate = "0"+str(row+1)
        melt_rate = melt_labels[row].replace("0.0","")   # e.g. from 0.001 to 01, or from 0.02 to 2
        file_number = str(int(t/(int(melt_rate)))).zfill(5)  # normalise t by the melt rate.  .zfill(5) fills the string with 0 until it's 5 characters long
        for x,x_val in enumerate(x_variable):
            if viscosity:
                exp = x_val.split('e')[-1] # the exponent after visc_ and before 5e3 or 1e4 etc
                # poro_file = 'wd05_visc/visc_'+exp+'_'+x_val+'/vis'+x_val+'_mR_0'+str(melt_rate)+'/a_porosity_'+file_number+'.png' 
                # bb_file = 'wd05_visc/visc_'+exp+'_'+x_val+'/vis'+x_val+'_mR_0'+str(melt_rate)+'/a_brokenBonds_'+file_number+'.png'
                # poro_file = 'visc_'+exp+'_'+x_val+'/vis'+x_val+'_mR_'+melt_rate+'/a_porosity_'+file_number+'.png' 
                poro_file = 'visc_'+exp+'_'+x_val+'/vis1e2_mR_'+melt_rate+'/a_porosity_'+file_number+'.png' 
                # poro_file = 'vis'+x_val+'_mR_'+melt_rate+'/a_porosity_'+file_number+'.png' 
                # bb_file = 'visc_'+exp+'_'+x_val+'/vis'+x_val+'_mR_'+melt_rate+'/a_brokenBonds_'+file_number+'.png'
                bb_file = 'visc_'+exp+'_'+x_val+'/vis1e2_mR_'+melt_rate+'/a_brokenBonds_'+file_number+'.png'
                # bb_file = 'vis'+x_val+'_mR_'+melt_rate+'/a_brokenBonds_'+file_number+'.png'
            elif def_rate:
                poro_file = 'thdef'+x_val+'/vis1e2_mR_'+melt_rate+'/a_porosity_'+file_number+'.png' # .zfill(5) fills the string with 0 until it's 5 characters long
                bb_file  = 'thdef'+x_val+'/vis1e2_mR_'+melt_rate+'/a_brokenBonds_'+file_number+'.png'

            if os.path.isfile(poro_file):
                poro_big_file = Image.open(poro_file)  # I open it here so then I can call poro_big_file.close() and close it
                poro = np.asarray(poro_big_file.crop((585,188,1468,1063)).convert('RGB'))
                big_array[row*cols+2*x,:,:,:] = poro  # 0 2 4 rows*2 because there are 2 images for each simulation. First index is for each image
                #print(row*cols+2*x)
                poro_big_file.close()
                # print(f'file exists {poro_file}')
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
        print("mrate ",melt_rate)
        # end of viscosity/deformation loop
        melt_labels_t[row] = melt_labels[row]+" ("+file_number+")"
        print(melt_labels_t[row])


    
    return big_array,melt_labels_t  # the number of columns is the num of x variable times two (two images for each sim)
    # return np.array([bb,poro,bb,poro])

ncols = len(x_variable)*2
for t in times:
    # array,file_numbers = make_array(x_variable,melt_labels,t)   # load all files
    array,melt_and_time = make_array(x_variable,melt_labels,t)   # load all files
    result = gallery(array,ncols)           # organise in gallery view
    # plt.figure()  # need this otherwise the figures after the first one are tiny
    fig, ax = plt.subplots()
    ax.imshow(result.astype('uint8')) # uint8 explanation: https://stackoverflow.com/questions/49643907/clipping-input-data-to-the-valid-range-for-imshow-with-rgb-data-0-1-for-floa
    ax.set_xticks(np.arange(437,875*len(x_variable)*2,875*2)) #  position and values of ticks on the x axis. Start: 437 (half width of an image) End: length of image times num of images in one row, Step: 883 (legth of image)
    ax.set_xticklabels(x_variable,fontsize=4)

    if viscosity:
        ax.set_xlabel('Viscosity')
    elif def_rate:
        ax.set_xlabel('Deformation rate')

    # plt.yticks(np.arange(441,883*4,883), ['0.01','0.02','0.03','0.04'])
    y_ticks_positions = np.arange(441,883*len(melt_labels)+441,883)
    ax.set_yticks(y_ticks_positions)
    ax.set_yticklabels(melt_and_time,fontsize=4)   #  position and values of ticks on the y axis. Start: 441 (half of image height) End: height of image times num of images in one column, Step: 883 (height of image)
    ax.set_ylabel('Melt increment per spot')
    ax.set_title("$t_{ref}$ = "+str(t),fontsize=8)
    # ax.set_ylim([0,883*len(melt_labels)*1.05])  # height * n of images plus a bit more, no clue why but it kinda works

    # # create a second axis for y to add a second set of labels on the right hand side (time - file number)
    # y2 = ax.twinx()
    # # y2.set_ylim(ax.get_ylim()) # Copy the y limits from the left axis. Need it because y axis increases from top to bottom. Otherwise it's flipped

    # y2.set_yticks(~y_ticks_positions)  #same as above, but reverse order
    # y2.set_yticklabels(file_numbers) # Set the right y-tick labels
    # y2.set_ylabel("File number")
    # print(ax.get_yticks())
    # print(y2.get_yticks())
    plt.tight_layout()
    # plt.savefig('visc_images/wd_visc_mRate_t'+str(t).zfill(3)+'.png',dpi=1200)
    # plt.savefig('images_in_grid/wd_dRate_mRate_t'+str(t).zfill(3)+'.png',dpi=1200)
    if not os.path.exists('images_in_grid'):
        os.makedirs('images_in_grid')
    plt.savefig('images_in_grid/visc_mRate_t'+str(t).zfill(3)+'.png',dpi=600)
    
    # plt.show()

    plt.clf()   # close figure
