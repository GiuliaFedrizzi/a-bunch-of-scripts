import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
from pathlib import Path


# import functions from external file
from useful_functions import * 

average_porosity = []
average_porosity_meltArea = []
time_array = []

input_tstep = float(getTimeStep("input.txt"))
for i,filename in enumerate(sorted(glob.glob("my_experiment*"))[0:60:1]): #[beg:end:step]  set which timesteps (based on FILE NUMBER) 
    myfile = Path(os.getcwd()+'/'+filename)
    if myfile.is_file():
        myExp = pd.read_csv(filename, header=0)
        # average everywhere
        average_porosity.append(myExp['Porosity'].mean())
        # average in the melting area only
        average_porosity_meltArea.append(myExp[myExp["y coord"]<0.2]["Porosity"].mean())  # mean with conditions on x and y coordinates
        file_num = float(filename.split("experiment")[1].split(".")[0])  # first take the part after "experiment", then the one before the "."
        time_array.append(file_num*input_tstep)

# calculate rate of porosity increase (final poro - initial poro over total time)
rate = (time_array[-1]-time_array[0])/(average_porosity[-1]-average_porosity[0])
# print(average_porosity[-1]-average_porosity[0])
print("rate = ",str(rate))

plt.scatter(time_array,average_porosity,label='Everywhere')
plt.scatter(time_array,average_porosity_meltArea,label='Melting area')
plt.xlabel("Time (s)")
plt.ylabel("Average Porosity")
plt.legend()
plt.title(os.getcwd())
plt.show()

