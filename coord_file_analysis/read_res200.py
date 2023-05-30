"""
read only coordinates from res200.elle, the input file
it reads the "UNODES" section of the file
calculates the difference in y coordinate between two consecutive rows
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np

os.chdir('/home/home01/scgf/myscripts/coord_file_analysis/')
write_file = False  # option to write out a new res200 file
plot_diff = True    # option to plot the difference in y coordinate between 2 rows 

file_name = "/nobackup/scgf/myExperiments/gaussJan2022/gj169/baseFiles/res200.elle"

if write_file:
    with open(file_name) as input_file:
        head = [next(input_file) for _ in range(12457)]  # save the first part 

    with open("res200new.elle", 'w') as f:
        f.write("".join(head))
    #    f.write("".join(coord_matrix))
    #    f.write("\n")
        f.close()

df = pd.read_csv(file_name,skiprows=12457,delim_whitespace=True,header=None)  # read as dataframe
# first line = 12458 last = end of file

df.columns = ['id_n', 'x', 'y', 'conc']    # rename columns

df.drop(columns=['conc'], inplace=True)   # drop the first column (duplicate) and the conc column

y_res200 = df[0:200:1].values

# extract values along a vertical line and calculate the difference between one and the next (vertically)
df_v = df.copy(deep=True)
df_v = df_v[50:len(df):200]    # dataframe only containing a vertical line. start from the 50th element and skip a row of 200 elements
ycoord_vals = df_v["y"].values
diff_1 = list(ycoord_vals[1:] - ycoord_vals[:-1])  # calculate the difference between 2 consecutive elements
diff_1.append(diff_1[-1])  # copy the last value to reach the same length
df_v["diff"] = diff_1  # add a column to the dataframe
y_array = np.arange(0,len(df_v))  # to plot the difference
print(ycoord_vals)
print(df_v["diff"].values)


#  calculate the new values - the more accurate ones
h = (1/200)*np.sqrt(3)/2  # one height of a triangle with edge = (1/200)
n_id = 0    # initialise index
if write_file:
    with open("res200new.elle", 'a')as f:
        for i in range(0,230):
            ny = h/2 + i*h   # build the new values of y  
            ny_200 = np.full(200, ny)
            # print(ycoord_vals[i],ny)  # check that they are the same (well, that they are close)
            for j in range(0,200):
                if i%2 == 0:    # even rows
                    nx = j/200
                else:
                    nx = (j+0.5)/200   # odd rows are shifted by half a triangle edge
                # if i == 1:
                #     print(nx) 
                # print(str(n_id),str(nx),str(ny),str(1))
                line = [str(n_id)," ",str(nx)," ",str(ny)," ",str(1),"\n"]  # convert to strings, add spaces and new line at the end

                for l in line:
                    f.write("%s" % l)

                n_id+=1  # increment the counter for the particle id
                
    f.close()

if plot_diff:
    fig, ax1 = plt.subplots(nrows=1,ncols=1)

    # sns.scatterplot(data=df_v ,x=df_v.index,y='diff',ax=ax1)
    # sns.lineplot(data=df_v ,x=df_v.index,y='diff',ax=ax1)
    ax1.plot(df_v['diff'].values,y_array)
    ax1.scatter(df_v['diff'].values,y_array)

    ax1.get_xaxis().get_major_formatter().set_useOffset(False)  # avoid weird axis formatting
    # sns.lineplot(data=df ,x=df.index,y='y')
    plt.show()
