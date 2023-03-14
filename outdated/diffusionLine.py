""" plot a line for pressure diffusion. From a single directory.
 only every x timesteps (files) """

# I use this ^ to run python in VS code in interactive mode
import pandas as pd
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy import interpolate
import os
import glob
import seaborn as sns

#os.chdir('/nobackup/scgf/myExperiments/gaussTimeOnce/gaussTimeOnce200/sigma_1_0/sigma_1_0_gaussTime05/tstep04_4_5e4/')
filename = 'my_experiment00080.csv'

def getTimeStep(inputFile):
    with open(inputFile) as iFile:
        for num, line in enumerate(iFile,1):
            if "Tstep" in line:
                input_tstep = line.split(" ")[1]  # split before and after space, take the second word (value of timestep)
                return input_tstep  

# open the first file after melt addition to find the max P
myExp = pd.read_csv(filename, header=0)
#xmax = myExp.loc[myExp['Pressure'].idxmax(), 'x coord']
#ymax = myExp.loc[myExp['Pressure'].idxmax(), 'y coord']
xmax = 0.505
ymax = 0.505

# find the x coordinates that corresponds to the max pressure
x_array = np.array(myExp.loc[myExp.apply(lambda x: math.isclose(x['y coord'],ymax,rel_tol=1e-3),axis=1),'x coord'])

# get the y coordinates of all x in range of tolerance
y_array = np.array(myExp.loc[myExp.apply(lambda x: math.isclose(x['x coord'],xmax,rel_tol=1e-3),axis=1),'y coord'])

fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2)
# zoom in. Limits are max location +- 0.05
ax1.set_xlim([xmax-0.05,xmax+0.05])
ax2.set_xlim([ymax-0.05,ymax+0.05])

count=0

for i,filename in enumerate(sorted(glob.glob("my_experiment*"))):
    if i == 0:
         continue  # skip t = 0
    if i%100 == 0:   #  plot only every x timesteps (files)
        myExp = pd.read_csv(filename, header=0)
        
        pressure_array_x = np.array(myExp.loc[myExp.apply(lambda x: math.isclose(x['y coord'],ymax,rel_tol=1e-3),axis=1),'Pressure'])
        pressure_array_y = np.array(myExp.loc[myExp.apply(lambda x: math.isclose(x['x coord'],xmax,rel_tol=1e-3),axis=1),'Pressure'])

        input_tstep = float(getTimeStep("input.txt"))
        file_num = float(filename.split("experiment")[1].split(".")[0])  # first take the part after "experiment", then the one before the "."
        # name of the line
        labelName = "t=" + str('{:.1e}'.format(input_tstep*file_num))

        plotStyle='-'

        # ax1.plot(x_array, pressure_array_x,plotStyle,label=labelName)
        # ax1.legend()
        ax1.set_ylabel("Fluid Pressure")
        ax1.set_title("Horizontal Profile")

        ax2.plot(y_array, pressure_array_y,plotStyle,label=labelName)
        #ax2.legend()
        ax2.set_title("Vertical Profile")

        data = {'x_coord': x_array,'pressure': pressure_array_x}
        df = pd.DataFrame(data)
        sns.lineplot(data=df,x="x_coord",y="pressure",ax=ax1)
        ax1.set_ylabel("Fluid Pressure")
        ax1.set_title("Horizontal Profile")



fig.suptitle(os.getcwd().split("myExperiments/")[1]) # get the part of the path that is after "myExperiments/"

# LEGEND:
# get the position of the plot so I can add the legend to it 
box = ax2.get_position()
#box = ax1.get_position()


# upper left = the point that I am setting wiht the second argument
#ax3.legend(loc='upper left',bbox_to_anchor=(0,-0.1),fancybox=True, ncol=8)
ax2.legend(loc='center left',bbox_to_anchor=(1,0.5),fancybox=True, ncol=1) 
#plt.tight_layout()
plt.show()
#plt.savefig("porDiff01_a.png", dpi=150,transparent=False)
"""
# %% porosity distribution
poroRead = pd.read_csv("my_experiment019.csv", header=0)

porosityAll = poroRead['Porosity'].tolist()

porosityAll.sort()
xArrayPoro=list(range(1, len(porosityAll)+1))

plt.plot(xArrayPoro,porosityAll)
# %%
"""
