import os
import datetime
import matplotlib.pyplot as plt
from itertools import accumulate

# Function to convert timestamp to datetime
def get_creation_time(file_path):
    timestamp = os.path.getctime(file_path)
    return datetime.datetime.fromtimestamp(timestamp)

fig, ax = plt.subplots(2, 2, figsize=(9, 6))
ax = ax.flatten()

prt_dirs = ['prt06','prt07']
second_part_of_path = 'rt0.001/visc_4_1e4/vis1e4_mR_01'
time_diffs_all = []

for i,prt_dir in enumerate(prt_dirs):
    directory = '/nobackup/scgf/myExperiments/threeAreas/prod/prt/'+prt_dir+"/"+second_part_of_path

    # Get list of files that match the pattern 'my_experiment*.csv'
    file_list = [f for f in os.listdir(directory) if f.startswith('my_experiment') and f.endswith('.csv')]
    file_list.sort()  # Sort the files by name

    # Get the creation times for each file
    creation_times = [get_creation_time(os.path.join(directory, file)) for file in file_list]
    # Calculate the time differences in minutes
    time_diffs = [(creation_times[i] - creation_times[i - 1]).total_seconds() / 60 for i in range(1, len(creation_times))]
    time_diffs = list(accumulate(time_diffs))
    time_diffs_all.append(time_diffs)

    # Plotting
    ax[0].plot(time_diffs, marker='o',label=prt_dir)
ax[0].set_title("Time Elapsed Between Consecutive Files")
ax[0].set_xlabel("File Index")
ax[0].legend()
ax[0].set_ylabel("Time Elapsed (minutes), cumulative")
ax[0].grid(True)
# plt.show()

# print(f'time_diffs_all {(time_diffs_all[0]))}')

improvement = [(a - b) for a, b in zip(time_diffs_all[0], time_diffs_all[1])]  # this works even if one list is longer (extra elements are ignored)

# plt.figure(figsize=(10, 6))

ax[1].plot(improvement, marker='o')
ax[1].set_title("Time improvement")
ax[1].set_xlabel("File Index")
ax[1].set_ylabel("Improvement over time")
ax[1].grid(True)
# plt.show()


improvement = [(a - b)/a for a, b in zip(time_diffs_all[0], time_diffs_all[1])]  # this works even if one list is longer (extra elements are ignored)

# plt.figure(figsize=(10, 6))

ax[2].plot(improvement, marker='o')
ax[2].set_title("Time improvement ratio")
ax[2].set_xlabel("File Index")
ax[2].set_ylabel("Improvement/Original time")
ax[2].grid(True)

fig.suptitle(second_part_of_path, fontsize=16)

plt.show()