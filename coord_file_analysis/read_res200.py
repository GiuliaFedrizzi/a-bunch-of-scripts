"""
read only coordinates from res200.elle, the input file
it reads the "UNODES" section of the file
calculates the difference in y coordinate between two consecutive rows
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

os.chdir('/home/home01/scgf/myscripts/coord_file_analysis/')
    
file_name = "res200.elle"

df = pd.read_csv(file_name,skiprows=12457,delim_whitespace=True,header=None)  # read as dataframe
# first line = 12458 last = end of file

df.columns = ['id_n', 'x', 'y', 'conc']    # rename columns

df.drop(columns=['id_n','conc'], inplace=True)   # drop the first column (duplicate) and the conc column

y_res200 = df[0:200:1].values

# extract values along a vertical line and calculate the difference between one and the next (vertically)
df_v = df.copy(deep=True)
df_v = df_v[50:len(df):200]    # dataframe only containing a vertical line. start from the 50th element and skip a row of 200 elements
ycoord_vals = df_v["y"].values
diff_1 = list(ycoord_vals[1:] - ycoord_vals[:-1])  # calculate the difference between 2 consecutive elements
diff_1.append(diff_1[-1])  # copy the last value to reach the same length
df_v["diff"] = diff_1  # add a column to the dataframe


#  calculate the new values - the more accurate ones
h = (1/200)*np.sqrt(3)/2  # one height of a triangle with edge = (1/200)
long_ny = []  # array with all values of new y, repeated 200 times (because x will vary)
n_id = 0    # initialise index

for i in range(0,230):
    ny = h/2 + i*h   # build the new values of y  
    ny_200 = np.full(200, ny)
    #long_ny.extend(ny_200)

    # long_ny.append(np.full(200, ny))  # repeat that value 200 times and append it to the previous array of y values
    # print(ycoord_vals[i],ny)  # check that they are the same (well, that they are close)
    for j in range(0,200):
        if i%2 == 0:    # even rows
            x = j/200
        else:
            x = (j+0.5)/200   # odd rows are shifted by half a triangle edge
        if i == 1:
            print(x) 
        n_id+=1

# print(len(long_ny))
# print(long_ny)
    

#print(diff_1)
# print(df["y"].values)

if False:
    fig, ax1 = plt.subplots(nrows=1,ncols=1)

    sns.scatterplot(data=df_v ,x=df_v.index,y='diff',ax=ax1)
    sns.lineplot(data=df_v ,x=df_v.index,y='diff',ax=ax1)
    ax1.get_yaxis().get_major_formatter().set_useOffset(False)
    # sns.lineplot(data=df ,x=df.index,y='y')
    plt.show()