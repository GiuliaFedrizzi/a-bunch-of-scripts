# %%
# here we go ...

# I use this ^ to run python in VS code in interactive mode
import pandas as pd
from matplotlib import colors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import math
from scipy import interpolate
import os
import glob
import cProfile
import pstats
profile = cProfile.Profile()
import re

filefrequency = 5


# plt.rcParams["font.weight"] = "bold"
# plt.rcParams['lines.linewidth'] = 2.5
# plt.rcParams['axes.linewidth'] = 2

big_font = 15

def getMaxPressures(myExp):
    # look for the max P and save its x and y coordinates
    maxP = max(myExp['Pressure'])
    #print("max fluid P: "+str(maxP)/1e6)#+" MPa in "+str(xmax)+", "+str(ymax))
    #print("max fluid P: "+str('{:.4e}'.format(max(myExp['Pressure'])))+" in "+str(xmax)+", "+str(ymax))
    return maxP

def getTimeStep(inputFile):
    with open(inputFile) as iFile:
        for num, line in enumerate(iFile,1):
            if "Tstep" in line:
                input_tstep = line.split(" ")[1]  # split before and after space, take the second word (value of timestep)
                return input_tstep  

# Get the number from the filename
def extract_number(filename):
    match = re.search(r"(\d+)", filename)
    if match:
        return int(match.group(1))
    return 0

fig, axs = plt.subplots(nrows=1,ncols=1)

#   plot broken bonds
# ax2 = axs.twinx()  # create a secon axis so I can plot broken bonds on the same figure
marks=[x for x in Line2D.markers]
# print(f'marks {marks[0]}, {marks[1]}')

for a,dir in enumerate(sorted(glob.glob("rt0.03/mrate3e6"))):
    os.chdir(dir)
    print(os.getcwd())
    maxPressure = [] # initialise empty array
    tot_bb = [] # initialise empty array
    time_array = []
    input_tstep = float(getTimeStep("input.txt"))

    i = 0
    for filename in (sorted(glob.glob("my_experiment*"),key=extract_number))[0::filefrequency]:
        if i == 0:
            i +=1 
            continue  # skip t = 0
        myExp = pd.read_csv(filename, header=0)
        maxPressure.append(getMaxPressures(myExp)/1e6)
        tot_bb.append(myExp['Broken Bonds'].sum())
        file_num = float(filename.split("experiment")[1].split(".")[0])  # first take the part after "experiment", then the one before the "."
        time_array.append(input_tstep*file_num)
        # # name of the line
        # labelName = "t=" + str('{:.1e}'.format(input_tstep*file_num))
        # stopping criterion otherwise it could load too many files
        i +=1 
        print(i)
        if i == 13*filefrequency:
            print(f'Stopping early')
            break
    print(f'Last plotted file: {filename}')
    plotStyle='-'+marks[a+2]
    plotStyle1='--'+marks[a+2]
    dirName = os.getcwd().split("/")[-1] + "/" + os.getcwd().split("/")[-2] # get the last part of the path 
    dirPress = os.getcwd().split("/")[-1] # get the part of the path with pressure increase
    # sigma_components = dirName.split("/")[1].split("_")   # take the second part after / in "gaussTime/sigma_1_0/sigma_1_0_gaussTime05/tstep04_5_1e5", split around "_"
    # labelName = "$\sigma$ = " + sigma_components[1] + "." + sigma_components[2]   # from sigma_1_0 to 1.0
    axs.plot(time_array, maxPressure,plotStyle,color='k',markersize=5,alpha=0.7)    #  label=dirName+", pressure",
    #  plot broken bonds:
    # ax2.plot(time_array, tot_bb,plotStyle1,label=dirName+", Broken Bonds",markersize=5,alpha=0.5)     
    # axs.plot([time_array[0],time_array[-1]], [maxPressure[0],maxPressure[-1]],'k--',alpha=0.7,label="line between first and last point")     
    os.chdir("../..")   
           

axs.set_xlabel("Time (s)",fontsize=big_font)
axs.set_ylabel("Max Fluid Pressure (MPa)",fontsize=big_font)

#  plot broken bonds:
# ax2.set_ylabel("Broken Bonds")
# axs.set_title("Max Fluid Pressure in time. Melt increment = " + dirPress.split("/")[-1] + "\nValues every " + str(filefrequency) + " files.")
#axs.xaxis.set_major_formatter(FormatStrFormatter('% .1e'))
#fig.suptitle(os.getcwd().split("myExperiments/")[1]) # get the part of the path that is after "myExperiments/"

# LEGEND:
# axs.legend(fancybox=True, ncol=1,loc='center right',bbox_to_anchor=(0.99,0.2)) 
#  plot broken bonds:
# ax2.legend(fancybox=True, ncol=1,loc='center right',bbox_to_anchor=(0.99,0.1)) 
axs.grid(linestyle='--',alpha=0.6)#(linestyle='-', linewidth=2)
#axs.legend()
plt.tight_layout()
# plt.show()

fig_name = "max_Pf_rt0.03_mrate3e6.png"
plt.savefig(fig_name,dpi=600)
