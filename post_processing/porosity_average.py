import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
from pathlib import Path
import re


# import functions from external file
from useful_functions import * 

average_porosity = []
average_porosity_fast = []
average_porosity_slow = []
time_array = []

def get_order(file):
    file_number = int(re.findall(r'\d+',file)[0])
    return file_number

sortedInputFiles = sorted(glob.glob('my_experiment*.csv'), key=get_order)   # order files based on their full number, found with get_order()

input_tstep = float(getTimeStep("input.txt"))
for i,filename in enumerate(sortedInputFiles[0:20:1]): #[beg:end:step]  set which timesteps (based on FILE NUMBER) 
    myfile = Path(os.getcwd()+'/'+filename)
    if myfile.is_file():
        myExp = pd.read_csv(filename, header=0)
        # average everywhere
        average_porosity.append(myExp['Porosity'].mean()*100)
        # average in the fast melting area only    THICK LAYERS
        fast1 = (myExp[(myExp["y coord"]>0.25) & (myExp["y coord"]<0.35)]["Porosity"])
        fast2 = (myExp[(myExp["y coord"]>0.65) & (myExp["y coord"]<0.75)]["Porosity"])
        # fast3 = (myExp[(myExp["y coord"]>0.75) & (myExp["y coord"]<0.81)]["Porosity"])

        # slow areas only:
        slow1 = (myExp[(myExp["y coord"]>0.04) & (myExp["y coord"]<0.15)]["Porosity"])
        slow2 = (myExp[(myExp["y coord"]>0.45) & (myExp["y coord"]<0.55)]["Porosity"])
        slow3 = (myExp[(myExp["y coord"]>0.85) & (myExp["y coord"]<0.95)]["Porosity"])
        # slow4 = (myExp[(myExp["y coord"]>0.90) & (myExp["y coord"]<0.95)]["Porosity"])


        # average_porosity_fast.append(myExp[(myExp["y coord"]>0.714) & (myExp["y coord"]<0.857)]["Porosity"].mean()*100)  # mean with conditions on x and y coordinates
        # average_porosity_fast.append(pd.concat([fast1, fast2, fast3]).mean()*100)  # mean with conditions on x and y coordinates
        average_porosity_fast.append(pd.concat([fast1, fast2]).mean()*100)  # mean with conditions on x and y coordinates
        # average_porosity_slow.append(myExp[(myExp["y coord"]>0.857) & (myExp["y coord"]<1.0)]["Porosity"].mean()*100)  # mean with conditions on x and y coordinates
        average_porosity_slow.append(pd.concat([slow1, slow2, slow3]).mean()*100)  # mean with conditions on x and y coordinates
        # average_porosity_slow.append(pd.concat([slow1, slow2, slow3, slow4]).mean()*100)  # mean with conditions on x and y coordinates
        file_num = float(filename.split("experiment")[1].split(".")[0])  # first take the part after "experiment", then the one before the "."
        time_array.append(file_num*input_tstep)

# calculate rate of porosity increase (final poro - initial poro over total time)
# rate = (time_array[-1]-time_array[0])/(average_porosity[-1]-average_porosity[0])
rate = (average_porosity[-1]-average_porosity[0])/(time_array[-1]-time_array[0])
rate_fast = (average_porosity_fast[-1]-average_porosity_fast[0])/(time_array[-1]-time_array[0])
rate_slow = (average_porosity_slow[-1]-average_porosity_slow[0])/(time_array[-1]-time_array[0])
print("global ",average_porosity[-1]-average_porosity[0],"%")
print("fast ",average_porosity_fast[-1]-average_porosity_fast[0],"%")
print("slow ",average_porosity_slow[-1]-average_porosity_slow[0],"%")

# print(time_array[-1]-time_array[0])

s_in_year = 60*60*24*365
print("rate = ",str(rate*s_in_year),"% per year")
print("1% every ",str(1/(rate*s_in_year))," years")
print("rate_fast = ",str(rate_fast*s_in_year),"% per year")
print("rate_slow = ",str(rate_slow*s_in_year),"% per year")
print(f'av rate from fast and slow {(rate_fast*3+rate_slow*4)/7*s_in_year}')
print("fast/slow: ",rate_fast/rate_slow)


plt.scatter(time_array,average_porosity,label='Everywhere')
plt.scatter(time_array,average_porosity_fast,label='Fast')
plt.scatter(time_array,average_porosity_slow,label='Slow')
plt.xlabel("Time (s)")
plt.ylabel("Average Porosity, %")
plt.legend()
plt.title(os.getcwd())
plt.show()

