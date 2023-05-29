"""
read only coordinates from res200.elle, the input file
it reads the "LOCATION" section of the file
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

os.chdir('/home/home01/scgf/myscripts/coord_file_analysis/')
    
file_name = "res200.elle"

df = pd.read_csv(file_name,skiprows=805,nrows=11649,delim_whitespace=True,header=None)  # read as dataframe
# df = pd.read_csv(file_name,skiprows=805,nrows=500,delim_whitespace=True,header=None)  # read as dataframe
# first line = 12458 last = end of file

df.columns = ['id_n', 'x', 'y']    # rename columns

df.drop(columns=['id_n'], inplace=True)   # drop the first column (duplicate)
# print(df)


if True:
    fig, ax1 = plt.subplots(nrows=1,ncols=1)
    plt.plot(df.x,df.y)
    plt.scatter(df.x,df.y)
    ax1.set_ylim(0.9,1.0)
    # sns.scatterplot(data=df ,x='x',y='y',ax=ax1)
    # sns.lineplot(data=df,x='x',y='y',ax=ax1)
    # ax1.get_yaxis().get_major_formatter().set_useOffset(False)
    plt.axis('equal')
    plt.show()
