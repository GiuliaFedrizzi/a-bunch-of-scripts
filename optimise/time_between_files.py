import os
import datetime
import matplotlib.pyplot as plt
from itertools import accumulate

# Function to convert timestamp to datetime
def get_creation_time(file_path):
    timestamp = os.path.getctime(file_path)
    return datetime.datetime.fromtimestamp(timestamp)

fig, ax = plt.subplots(2, 2, figsize=(9, 6))

# Adding subplots
ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
ax2 = plt.subplot2grid((2, 2), (0, 1))
ax3 = plt.subplot2grid((2, 2), (1, 1))

# prt_dirs = ['depth1000','depth5000']
# prt_dirs = ['threads1','threads2','threads4','threads8','threads16']#,'threads16']
prt_dirs = ['op13/rt0.03/threads1','op14/rt0.03/threads1','op13/rt0.03/threads8','op14/rt0.03/threads8','op17/rt0.03/threads8','op18/rt0.03/threads8']#,'threads16']
# second_part_of_path = 'scale1000/rb0.01'
time_diffs_all = []

for i,prt_dir in enumerate(prt_dirs):
    # directory = '/nobackup/scgf/myExperiments/threeAreas/prod/prt/'+prt_dir+"/"+second_part_of_path
    # directory = os.getcwd()+"/"+prt_dir+"/"+second_part_of_path
    directory = os.getcwd()+"/"+prt_dir#+"/rt0.08/mrate0.001"

    # Get list of files that match the pattern 'my_experiment*.csv'
    file_list = [f for f in sorted(os.listdir(directory))[:40] if f.startswith('my_experiment') and f.endswith('.csv')]
    file_list.sort()  # Sort the files by name
    # Get the creation times for each file
    creation_times = [get_creation_time(os.path.join(directory, file)) for file in file_list]
    # Calculate the time differences in minutes
    time_diffs = [(creation_times[i] - creation_times[i - 1]).total_seconds() / 60 for i in range(1, len(creation_times))]
    time_diffs = list(accumulate(time_diffs))
    time_diffs_all.append(time_diffs)

    # Plotting
    ax1.plot(range(0,len(time_diffs)),time_diffs, marker='o',label=prt_dir)
ax1.set_title("Time Elapsed Between Consecutive Files")
ax1.set_xlabel("File Index")
ax1.legend()
ax1.set_ylabel("Time Elapsed (minutes), cumulative")
ax1.grid(True)

# print(f'time_diffs_all {(time_diffs_all[0]))}')

improvement = [(a - b) for a, b in zip(time_diffs_all[0], time_diffs_all[2])]  # this works even if one list is longer (extra elements are ignored)


ax2.plot(improvement, marker='o')
ax2.set_title("Time improvement")
ax2.set_xlabel("File Index")
ax2.set_ylabel("Improvement over time")
ax2.grid(True)


improvement = [(a - b)/a for a, b in zip(time_diffs_all[0], time_diffs_all[2])]  # this works even if one list is longer (extra elements are ignored)


ax3.plot(improvement, marker='o')
ax3.set_title("Time improvement ratio")
ax3.set_xlabel("File Index")
ax3.set_ylabel("Improvement/Original time")
ax3.grid(True)

fig.suptitle(os.getcwd())

fig.tight_layout()
plt.show()