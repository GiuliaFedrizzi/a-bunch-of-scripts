# %%
# here we go ...

# I use this ^ to run python in VS code in interactive mode

# plots the time of the first broken bond
import matplotlib.pyplot as plt
import glob
import os as os

#parent_directory = "sigma_3_0_gaussTime05"
parent_directory = "timestep04"


fig, axs = plt.subplots(nrows=1, ncols=1)


#fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2)
#fig, ax1 = plt.subplots(nrows=1,ncols=1)

#ax3.set_xlim([xmax-0.05,xmax+0.05])
#ax4.set_xlim([ymax-0.05,ymax+0.05])
#ax3.set_ylim([0.054,0.055])

broken_bond_string = "Broken Bonds"
time_string = "time is "

def getFirstBrokenBond(myExp):
    time_line = 0
    with open(myExp) as expFile:
        for num, line in enumerate(expFile,1):
            if time_string in line:
                time_line = num  # gets overwritten every time until I find a broken bond. Then the value stays
                time_line_whole = line
            if broken_bond_string in line:  # at the FIRST occurrence of Broken Bonds, stop the search
                x = time_line_whole.split("time is ") # split the line with the info about time around the string "time is"
                timebb = x[1]    # take the second part, which is the one that comes after the string "time is"
                timebb = timebb.replace("\n","")  # get rid of \n
                return timebb  
    

# change directory (I want to have 1 script that I keep outside of the many directories)
os.chdir(parent_directory)
dirList = []
for my_dir in sorted(glob.glob("tstep*/")):
    """ get the list of directories"""
    dirList.append(my_dir)

timesteps = []
times_of_bb = []   # list of times at which we have the first broken bond
#for myDir in dirList[1:4]:
for myDir in dirList:
    # loop through directories
    os.chdir(myDir)
    #print("--> Entering " + str(os.getcwd()))
    x = str(myDir).split("_")  # split where there are underscores
    timestep = x[2]  # take the third part of the string, e.g. 1e4
    try:
        time_of_first_bb = getFirstBrokenBond("latte.log")
    except:
        print("couldn't open the file.")
    if time_of_first_bb:
        timesteps.append(float(timestep.replace("/","")))
        times_of_bb.append(float(time_of_first_bb))

    os.chdir("..")

axs.plot(timesteps,times_of_bb, '--o',linewidth=2)
axs.set_xscale('log')
fig_title = "Time of the first broken bond using different time steps\n" #+ parent_directory.replace("/","-") 
axs.set(title=fig_title, xlabel='Time step (s)', ylabel='Time before the first broken bond (s)')
#axs.set_xscale('log')
figure_name = parent_directory.replace("/","-")+"_time_first_bb_aug2022.png"
plt.savefig(figure_name, dpi=600)
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
