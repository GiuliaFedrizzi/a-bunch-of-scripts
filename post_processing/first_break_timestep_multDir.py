#!/home/home01/scgf/.conda/envs/condaPandas/bin/python3
# %%
# here we go ...

# I use this ^ to run python in VS code in interactive mode

# plots the time of the first broken bond
# goes through multiple directories (multiple sigmas)
# to be run from the parent directory of all sigmas
import matplotlib.pyplot as plt
import glob
import os as os
from useful_functions import getFirstBrokenBond

parent_directories = []
for sigma_dir in sorted(glob.glob("sigma*/")):
    """ get the list of directories """
    parent_directories.append(sigma_dir)
# parent_directories = ["sigma_3_0"]
fig, axs = plt.subplots(nrows=1, ncols=1)

#ax3.set_xlim([xmax-0.05,xmax+0.05])
#ax4.set_xlim([ymax-0.05,ymax+0.05])
#ax3.set_ylim([0.054,0.055])

broken_bond_string = "Broken Bonds"
time_string = "time is "

# def getFirstBrokenBond(myExp):
#     time_line = 0
#     with open(myExp) as expFile:
#         for num, line in enumerate(expFile,1):
#             if time_string in line:
#                 time_line = num  # gets overwritten every time until I find a broken bond. Then the value stays
#                 time_line_whole = line
#             if broken_bond_string in line:  # at the FIRST occurrence of Broken Bonds, stop the search
#                 x = time_line_whole.split("time is ") # split the line with the info about time around the string "time is"
#                 timebb = x[1]    # take the second part, which is the one that comes after the string "time is"
#                 timebb = timebb.replace("\n","")  # get rid of \n
#                 return timebb  
    
for parent_directory in sorted(glob.glob("sigma*/")):   # sigma_*
    # change directory (I want to have 1 script that I keep outside of the many directories)
    os.chdir(parent_directory)
    os.chdir(parent_directory.replace("/","") + "_gaussTime05")
    print(os.getcwd)
    dirList = []
    for my_dir in sorted(glob.glob("tstep*/")):
        """ get the list of directories"""
        dirList.append(my_dir)
    print(os.getcwd())
    timesteps = []
    times_of_bb = []   # list of times at which we have the first broken bond
    #for myDir in dirList[1:4]:
    for myDir in dirList:
        # loop through directories
        os.chdir(myDir)  # get inside tstep02_*_1e*
        x = str(myDir).split("_")  # split where there are underscores
        timestep = x[2]  # take the third part of the string, e.g. 1e4
        try:
            time_of_first_bb = getFirstBrokenBond("latte.log")
        except:
            print("couldn't open the file.")
        if time_of_first_bb:   # if you found a broken bond...
            timesteps.append(float(timestep.replace("/","")))  # ...get the time
            times_of_bb.append(float(time_of_first_bb))


        os.chdir('..')
    my_label = "$\sigma$ = " + parent_directory.split("_")[1] + "." + parent_directory.split("_")[2].replace("/","")
    axs.plot(timesteps,times_of_bb, '--o',linewidth=2,label=my_label)  # ...and plot it
    axs.set_xscale('log')
    #axs.set_yscale('log')
    os.chdir("../..")
fig_title = "Time of the first broken bond for different time steps\n"
axs.set(title=fig_title, xlabel='Time step (s)', ylabel='Time before first broken bond (s)')
axs.grid(linestyle='--',alpha=0.6)#(linestyle='-', linewidth=2)
#axs.set_xscale('log')
#figure_name = parent_directory.replace("/","-")+"_time_first_bb.png"
#plt.savefig(figure_name, dpi=600)
plt.legend()
plt.show()
"""
# LEGEND:
# get the position of the middle plot so I can add the legend to it (all_axes[1].legend)
middle_subplot = int(tot_files/2)
#box = all_axes[middle_subplot].get_position()

box = all_axes[-1].get_position()

#fig.suptitle("Profiles along the point of Maximum Pressure")

# upper left = the point that I am setting wiht the second argument
#ax3.legend(loc='upper left',bbox_to_anchor=(0,-0.1),fancybox=True, ncol=8)

all_axes[-2].legend(loc='center left',bbox_to_anchor=(0,-0.5),fancybox=True, ncol=10) 
all_axes[middle_subplot].set_ylabel("Fluid Pressure")   # the middle plot (tot_files/2)
all_axes[-1].set_ylabel("Fluid Pressure")   # the middle plot (tot_files/2)
all_axes[0].set_title("Horizontal Profile")
all_axes[0].set_title("Vertical Profile")
"""

