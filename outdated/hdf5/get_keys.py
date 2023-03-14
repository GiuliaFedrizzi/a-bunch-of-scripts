"""
mini-script that 
1. copies the h5 file (so if it needs to be accessed it doesn't get locked)
2. reads and prints the h5 heys
"""

import pandas as pd
import shutil

file_name_root = "my_h5_limitFiles0_sigma3_par"  # the name before .h5, same for the original and the copy
orig_file = file_name_root + ".h5" # name of the original h5 file
copied_file = file_name_root + "_copy.h5"  # name of the copy

shutil.copy(orig_file, copied_file)  # save a copy

h5 = copied_file

store = pd.HDFStore(h5)
#store.walk_nodes
#my_h5 = tb.open_file(h5)
for key in store.keys():
    print(key)
#print(store.keys())
