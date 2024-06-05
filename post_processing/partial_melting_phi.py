"""
Visualise the effect of partial melting on the porosity:
the area of the particles gets smaller
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.rcParams["xtick.top"] = True
plt.rcParams["xtick.labeltop"] = True
plt.rcParams["xtick.bottom"] = False
plt.rcParams["xtick.labelbottom"] = False
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15

filename = 'my_experiment00000.csv'
def read_calculate_plot(filename):
    """
    read the csv file, plot
    """
    myExp_all = pd.read_csv(filename, header=0)

    mask = myExp_all['y coord'] < 0.55    # create a mask: use these values to identify which rows to keep
    myExp_all = myExp_all[mask].copy()  # keep only the rows found by mask

    mask = myExp_all['y coord'] > 0.46    # create a mask: use these values to identify which rows to keep
    myExp_all = myExp_all[mask].copy()  # keep only the rows found by mask

    mask = myExp_all['x coord'] < 0.55    # create a mask: use these values to identify which rows to keep
    myExp_all = myExp_all[mask].copy()  # keep only the rows found by mask

    mask = myExp_all['x coord'] > 0.46    # create a mask: use these values to identify which rows to keep
    myExp = myExp_all[mask].copy()  # keep only the rows found by mask

    myExp['solid_frac'] = 1-myExp['Porosity']
    # print(max(myExp['solid_frac']),min(myExp['solid_frac']))
    plt.figure(figsize=(7,7))
    # vmin=0, vmax=6
    sns.scatterplot(data=myExp,x="x coord",y="y coord",color='silver',linewidth=1.5,edgecolor='k',alpha=0.8,
        size="solid_frac",sizes=(50, 400)).set_aspect('equal') #legend='full'
    plt.xlabel('')
    plt.ylabel('')
    plt.legend([],[], frameon=False)
    # plt.show()
    plt.savefig("partial_melt_phi_circles.png")

myfile = Path(os.getcwd()+'/'+filename)  # build file name including path
if myfile.is_file():
    variable_hue = read_calculate_plot(filename)