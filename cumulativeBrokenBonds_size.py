#!/home/home01/scgf/.conda/envs/condaPandas/bin/python3

""" plots the cumulative number of broken bonds with increasing time
 Giulia Oct 2022
version for gaussShearFixed, where the 'inner' directories are called 'size020', 'size030', etc
The number of broken bonds can be normalised to account for the different size of a roken bond if 

"""
# goes through multiple directories
import matplotlib.pyplot as plt
import glob
import os as os
import sys

#timestep_dir = "tstep04_3_5e3"

dir_labels = ['020','030', '040','050','060','070','080','090','100']
resolution = 200

normalise=True

parent_directories = []
first_part_of_path = '/nobackup/scgf/myExperiments/gaussScaleFixFrac2/widthscaleOnly200/size'

for i in dir_labels:
    parent_directories.append(first_part_of_path+str(i))

fig, axs = plt.subplots(nrows=1, ncols=1)

broken_bond_string = "Broken Bonds"
time_string = "time is "

# get the scale from the path
scale_string = os.getcwd().split("/")[-1]

def getBrokenBondTime(lattefile):
    """ every time you find "Broken Bonds", save the number of bb present and the relative time"""
    time_line = 0
    time_bb_array = [];bb_array = []  # reset these two arrays
    with open(lattefile) as expFile:
        for num, line in enumerate(expFile,1):
            if time_string in line:
                time_line = num  # gets overwritten every time until I find a broken bond. Then the value stays
                time_line_whole = line   # save the whole line where you found  "time is "
            if broken_bond_string in line:  # every time you find "Broken Bonds", save the number of bb present and the relative time
                x = time_line_whole.split("time is ") # split the line with the info about time around the string "time is"
                timebb = x[1]    # take the second part, which is the one that comes after the string "time is"
                timebb = timebb.replace("\n","")  # get rid of \n
                broken_bond_number = line.split("Broken Bonds ")[1]  # take what comes after "Broken Bonds"
                broken_bond_number = broken_bond_number.split(",")[0]   # take what comes before the space and comma
                bb_array.append(float(broken_bond_number))  # build the array with the number of bb present at the time
                time_bb_array.append(float(timebb))              # build the array with the relative time 
    return time_bb_array,bb_array  
    
def plot_bb_in_time(parent_directory):
    # change directory (I want to have 1 script that I keep outside of the many directories)
    #try:
    time_bb_array,bb_array = getBrokenBondTime(parent_directory+"/latte.log")
    #except:
    #    print("Failed to get bb or time.")
    domain_size=parent_directory.split("/")[-1].replace("size","")
    my_label = "size = " + domain_size
    if normalise:
        bb_array_norm = [x / (float(domain_size)) for x in bb_array]  # number of broken bonds normalised by the domain scale
        axs.plot(time_bb_array,bb_array_norm, '--o',linewidth=1,markersize=1,label=my_label)  # ...and plot it
        axs.set(ylabel='Number of broken bonds (normalised)',ylim=[0,20])
    else:    
        axs.plot(time_bb_array,bb_array, '--o',linewidth=1,markersize=1,label=my_label)  # ...and plot it
        axs.set(ylabel='Number of broken bonds')
    #axs.set_xscale('log')
    #axs.set_yscale('log')

for parent_directory in parent_directories:   # sigma_*
    plot_bb_in_time(parent_directory)

fig_title = "Number of broken bonds (cumulative) in time\n " + first_part_of_path + "*"

axs.set(title=fig_title, xlabel='Time (s)')
axs.grid(linestyle='--',alpha=0.6)#(linestyle='-', linewidth=2)
# #axs.set_xscale('log')
# #figure_name = parent_directory.replace("/","-")+"_time_first_bb.png"
# #plt.savefig(figure_name, dpi=600)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.legend()
plt.show()

