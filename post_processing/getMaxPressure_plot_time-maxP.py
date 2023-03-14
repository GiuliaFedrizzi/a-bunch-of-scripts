# %%
# here we go ...

# I use this ^ to run python in VS code in interactive mode
import pandas as pd
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy import interpolate
import os
import glob
import cProfile
import pstats
profile = cProfile.Profile()

filefrequency = 100


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



fig, axs = plt.subplots(nrows=1,ncols=1)

for dir in sorted(glob.glob("sigma*/*gaussTime05/tstep04_5_1e5")):
    os.chdir(dir)
    print(os.getcwd())
    maxPressure = [] # initialise empty array
    time_array = []


    for i,filename in enumerate(sorted(glob.glob("my_experiment*"))):
        if i == 0:
            continue  # skip t = 0
        if i%filefrequency == 0:   #  plot only every x timesteps (files)
            myExp = pd.read_csv(filename, header=0)
            maxPressure.append(getMaxPressures(myExp))
            input_tstep = float(getTimeStep("input.txt"))
            file_num = float(filename.split("experiment")[1].split(".")[0])  # first take the part after "experiment", then the one before the "."
            time_array.append(input_tstep*file_num)
            # # name of the line
            # labelName = "t=" + str('{:.1e}'.format(input_tstep*file_num))
            if i == 20*filefrequency:
                break
    plotStyle='-o'
    dirName = os.getcwd().split("myExperiments/")[1] # get the part of the path that is after "myExperiments/"
    sigma_components = dirName.split("/")[1].split("_")   # take the second part after / in "gaussTime/sigma_1_0/sigma_1_0_gaussTime05/tstep04_5_1e5", split around "_"
    labelName = "$\sigma$ = " + sigma_components[1] + "." + sigma_components[2]   # from sigma_1_0 to 1.0
    axs.plot(time_array, maxPressure,plotStyle,label=labelName)     
    os.chdir("../../..")   
        

axs.set_xlabel("Time")
axs.set_ylabel("Max Fluid Pressure")
axs.set_title("Max Fluid Pressure in time. Time step = " + dirName.split("/")[-1].split("_")[-1] + "\nValues every " + str(filefrequency) + " files.")
#axs.xaxis.set_major_formatter(FormatStrFormatter('% .1e'))
#fig.suptitle(os.getcwd().split("myExperiments/")[1]) # get the part of the path that is after "myExperiments/"

# LEGEND:
# get the position of the plot so I can add the legend to it 
#box = axs.get_position()

                # upper left = the point that I am setting wiht the second argument
#axs.legend(loc='center left',bbox_to_anchor=(1,0.5),fancybox=True, ncol=1) 
#axs.legend(loc='upper left',fancybox=True, ncol=1) 
axs.legend(fancybox=True, ncol=1) 
axs.grid(linestyle='--',alpha=0.6)#(linestyle='-', linewidth=2)
#axs.legend()
#plt.tight_layout()
plt.show()
"""
# %% porosity distribution
poroRead = pd.read_csv("my_experiment019.csv", header=0)

porosityAll = poroRead['Porosity'].tolist()

porosityAll.sort()
xArrayPoro=list(range(1, len(porosityAll)+1))

plt.plot(xArrayPoro,porosityAll)
# %%
"""
