""" plot a line = pressure diffusion
 1st plot: horizontal, 2nd plot: vertical

can plot from different directories (e.g. sigma, resolution)
builds 2 pandas dataframes of the type:

      x_coord    pressure  dir      time
0     0.00125  40024800.0    1  t=1.0e+06
1     0.00375  40024800.0    1  t=1.0e+06
2     0.00625  40024800.0    1  t=1.0e+06
3     0.00875  40024800.0    1  t=1.0e+06
4     0.01125  40024800.0    1  t=1.0e+06
...       ...         ...  ...        ...
1195  0.98875  40024800.0    1  t=9.0e+06
1196  0.99125  40024800.0    1  t=9.0e+06
1197  0.99375  40024800.0    1  t=9.0e+06
1198  0.99625  40024800.0    1  t=9.0e+06
1199  0.99875  40024800.0    1  t=9.0e+06

and the second one for x_coord

"""

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

# dir_list = ['/nobackup/scgf/myExperiments/gaussTime/sigma_3_0/sigma_3_0_gaussTime05/tstep04_5_1e5',
#     '/nobackup/scgf/myExperiments/gaussScale/scale100/sigma_3_0/sigma_3_0_gaussTime05/tstep04_5_1e5',
#     '/nobackup/scgf/myExperiments/gaussScale/scale050/sigma_3_0/sigma_3_0_gaussTime05/tstep04_5_1e5']

# dir_list = ['/nobackup/scgf/myExperiments/gaussTime/sigma_3_0/sigma_3_0_gaussTime05/tstep04_5_1e5',
#      '/nobackup/scgf/myExperiments/gaussScale/scale200_r200/sigma_3_0/sigma_3_0_gaussTime05/tstep04_5_1e5']

sigma_str = "/sigma_3_0"
time_str = '5_1e5'
dir_list = ['/nobackup/scgf/myExperiments/gaussTimeOnce/gaussTimeOnce200' + sigma_str + sigma_str +'_gaussTime05/tstep04_' + time_str,
    '/nobackup/scgf/myExperiments/gaussTimeOnce/gaussTimeOnce050' + sigma_str + sigma_str + '_gaussTime05/tstep04_' + time_str]


#dir_labels = ['200','100','50']
# dir_labels = ['400','200']  # res400.elle, res200.elle
dir_labels = ['200','50']  # res400.elle, res200.elle

fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2)
first_file = 'my_experiment00100.csv'
#data = {'x_coord': x_array}
#df_x = pd.DataFrame(data)
df_x = pd.DataFrame()
df_y = pd.DataFrame()

def getTimeStep(inputFile):
    with open(inputFile) as iFile:
        for num, line in enumerate(iFile,1):
            if "Tstep" in line:
                input_tstep = line.split(" ")[1]  # split before and after space, take the second word (value of timestep)
                return input_tstep  

dirnum = 0
for dir in dir_list:
    """ loop through directories"""
    os.chdir(dir)
    dirnum+=1 # counter for directories
    dir_label = dir_labels[dirnum-1]   # get the label that corresponds to the directory

    # open the first file after melt addition to find the max P
    myExp = pd.read_csv(first_file, header=0)
    xmax = myExp.loc[myExp['Pressure'].idxmax(), 'x coord']
    ymax = myExp.loc[myExp['Pressure'].idxmax(), 'y coord']

    # find the x coordinates that corresponds to the max pressure
    x_array = np.array(myExp.loc[myExp.apply(lambda x: math.isclose(x['y coord'],ymax,rel_tol=1e-4),axis=1),'x coord'])

    # get the y coordinates of all x in range of tolerance
    y_array = np.array(myExp.loc[myExp.apply(lambda x: math.isclose(x['x coord'],xmax,rel_tol=1e-4),axis=1),'y coord'])

    #ax1.set_xlim([xmax-0.05,xmax+0.05])  # zoom in. Limits are max location +- 0.05
    #ax2.set_xlim([ymax-0.05,ymax+0.05])

    for i,filename in enumerate(sorted(glob.glob("my_experiment*"))[1:1001:300]): #[beg:end:step]
        myExp = pd.read_csv(filename, header=0)
        pressure_array_x = np.array(myExp.loc[myExp.apply(lambda x: math.isclose(x['y coord'],ymax,rel_tol=1e-4),axis=1),'Pressure'])
        pressure_array_y = np.array(myExp.loc[myExp.apply(lambda x: math.isclose(x['x coord'],xmax,rel_tol=1e-4),axis=1),'Pressure'])

        input_tstep = float(getTimeStep("input.txt"))
        file_num = float(filename.split("experiment")[1].split(".")[0])  # first take the part after "experiment", then the one before the "."
        # name of the line
        labelName = "t=" + str('{:.1e}'.format(input_tstep*file_num))

        # x                  shift array by the max value so that the maximum is at zero
        data1_x = {'x': x_array - xmax,'Fluid Pressure': pressure_array_x, 'scale': dir_label,'time': labelName}  # save temporarily
        df1_x = pd.DataFrame(data1_x)
        df_x = pd.concat([df_x,df1_x], ignore_index=True) # append to old one

        # y                  shift array by the max value so that the maximum is at zero
        data1_y = {'y': y_array - ymax,'Fluid Pressure': pressure_array_y, 'scale': dir_label,'time': labelName}  # save temporarily
        df1_y = pd.DataFrame(data1_y)
        df_y = pd.concat([df_y,df1_y], ignore_index=True) # append to old one

g_x = sns.lineplot(data=df_x ,x="x",y="Fluid Pressure",ax=ax1,hue='time',style='scale',alpha=0.5)
g_y = sns.lineplot(data=df_y ,x="y",y="Fluid Pressure",ax=ax2,hue='time',style='scale',alpha=0.5)
ax1.set_ylabel("Fluid Pressure")
ax1.set_title("Horizontal Profile")
ax1.set_xlim([-0.06,+0.06])  # zoom in. Limits are max location +- 0.05
ax2.set_xlim([-0.06,+0.06])  # zoom in. Limits are max location +- 0.05
#ax2.plot(y_array, pressure_array_y,plotStyle,label=labelName)
#ax2.legend()
ax2.set_title("Vertical Profile")

g_x.legend_.remove()
fig.suptitle("$\sigma$ = "+sigma_str.replace("/sigma_","").replace("_",".")+", tstep = " + time_str.split("_")[1]) # get the part of the path that is after "myExperiments/"

# LEGEND:
# get the position of the plot so I can add the legend to it 
box = ax2.get_position()

# upper left = the point that I am setting wiht the second argument
ax2.legend(loc='center left',bbox_to_anchor=(1,0.5),fancybox=True, ncol=1)   # legend for the vertical line plot
# save:
#os.chdir('/nobackup/scgf/myExperiments/gaussScale')
#plt.savefig("gaussScale50-100-200_diff_hrz-vrt.png", dpi=600,transparent=True)
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
