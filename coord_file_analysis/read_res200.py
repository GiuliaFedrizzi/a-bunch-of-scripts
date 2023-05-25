"""
read only coordinates from res200.elle, the input file
it reads the "UNODES" section of the file
"""

import os
import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

os.chdir('/home/home01/scgf/myscripts/coord_file_analysis/')
    
file_name = "res200.elle"

df = pd.read_csv(file_name,skiprows=12457,delim_whitespace=True,header=None)  # read as dataframe
# first line = 12458 last = end of file

new_names = ['id_n', 'x', 'y', 'conc']    # rename columns
df.columns = new_names

df.drop(columns=['id_n','conc'], inplace=True)   # drop the first column (duplicate) and the conc column

df = df[50:len(df):200]    # dataframe only containing a vertical line. start from the 50th element and skip a row of 200 elements
ycoord_vals = df["y"].values
diff_1 = list(ycoord_vals[1:] - ycoord_vals[:-1])
diff_1.append(diff_1[-1])
df["diff"] = diff_1

fig, ax1 = plt.subplots(nrows=1,ncols=1)

sns.scatterplot(data=df ,x=df.index,y='diff',ax=ax1)
sns.lineplot(data=df ,x=df.index,y='diff',ax=ax1)
ax1.get_yaxis().get_major_formatter().set_useOffset(False)
# sns.lineplot(data=df ,x=df.index,y='y')
plt.show()