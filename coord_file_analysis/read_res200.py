"""
read only coordinates from res200.elle, the input file
it reads the "UNODES" section of the file
"""

import os
import pandas as pd
import glob

os.chdir('/home/home01/scgf/myscripts/coord_file_analysis/')
    
file_name = "res200.elle"

df = pd.read_csv(file_name,skiprows=12457,delim_whitespace=True,header=None)  # read as dataframe
# first line = 12458 last = end of file

new_names = ['id', 'x', 'y', 'conc']    # rename columns
df.columns = new_names

df.drop(columns=['id','conc'], inplace=True)   # drop the first column (duplicate) and the conc column
print(df)
