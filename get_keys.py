"""
mini-script that 
1. copies the h5 file (so if it needs to be accessed it doesn't get locked)
2. reads and prints the h5 heys
"""

import pandas as pd
import shutil
shutil.copy('my_h5.h5', 'my_h5_copy.h5')

h5 = "my_h5_copy.h5"

store = pd.HDFStore(h5)

print(store.keys())
