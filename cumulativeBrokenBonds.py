#!/home/home01/scgf/.conda/envs/condaPandas/bin/python3

""" plots the cumulative number of broken bonds with increasing time
 Giulia June 2022
 To be run from gaussTime, goes through sigma*/sigma*_gaussTime05/tstep04_5_1e5
"""
# goes through multiple directories
import matplotlib.pyplot as plt
import glob
import os as os
from joblib import Parallel, delayed
import multiprocessing

timestep_dir = "tstep04_4_1e4"
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
    os.chdir(parent_directory)
    os.chdir(parent_directory.replace("/","") + "_gaussTime05")  # build the path
    print(os.getcwd())
    os.chdir(timestep_dir)  # get inside tstep04_... (chooses always the same timestep)
    #try:
    time_bb_array,bb_array = getBrokenBondTime("latte.log")
    #except:
    #    print("Failed to get bb or time.")

    os.chdir('..')
    my_label = "$\sigma$ = " + parent_directory.split("_")[1] + "." + parent_directory.split("_")[2].replace("/","")  # build the label sigma = x (from the directory name)
    axs.plot(time_bb_array,bb_array, '--o',linewidth=2,markersize=3,label=my_label)  # ...and plot it
    #axs.set_xscale('log')
    #axs.set_yscale('log')
    os.chdir("../..")

for parent_directory in parent_directories:   # sigma_*
    plot_bb_in_time(parent_directory)

fig_title = "Number of broken bonds (cumulative) in time for different $\sigma$ values\nTimestep = " + timestep_dir.split("_")[2] # take the third part of the string, e.g. 1e4
axs.set(title=fig_title, xlabel='Time (s)', ylabel='Number of broken bonds')
axs.grid(linestyle='--',alpha=0.6)#(linestyle='-', linewidth=2)
# #axs.set_xscale('log')
# #figure_name = parent_directory.replace("/","-")+"_time_first_bb.png"
# #plt.savefig(figure_name, dpi=600)
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
"""
for i,ax in enumerate(all_axes):
    # zoom in. Limits are max location +- 0.05
    if i%2 == 0: # even => horizontal
        all_axes[i].set_xlim([xmax-0.02,xmax+0.02])
    else:
        all_axes[i].set_xlim([ymax-0.02,ymax+0.02])
    all_axes[i].set_ylim([ymax-0.02,ymax+0.02])

#plt.tight_layout()
os.chdir('..')
plt.savefig(parent_directory+".png",dpi=600)
plt.show()

    

# -- end directory loop --

ax1.set_ylabel("Fluid Pressure")
ax1.set_title("Horizontal Profile")
ax2.set_title("Vertical Profile")

# zoom in. Limits are max location +- 0.05
ax1.set_xlim([xmax-0.05,xmax+0.05])
ax2.set_xlim([ymax-0.05,ymax+0.05])
# LEGEND:
# get the position of the 3rd plot so I can add the legend to it (ax3.legend)
box = ax2.get_position()

#fig.suptitle("Profiles along the point of Maximum Pressure")

# upper left = the point that I am setting wiht the second argument
#ax3.legend(loc='upper left',bbox_to_anchor=(0,-0.1),fancybox=True, ncol=8)
ax2.legend(loc='center left',bbox_to_anchor=(1,0.5),fancybox=True, ncol=1) 
#plt.tight_layout()
plt.show()

"""
