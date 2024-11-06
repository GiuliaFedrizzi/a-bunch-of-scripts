import numpy as np
import matplotlib.pyplot as plt
import json


def plot_options(ax,max_y_from_file):
    if max_y_from_file == 0:
        y_max = 1700 #535  # arbitrary number, the histograms are empty anyway
    else:
        y_max = max_y_from_file  # set a maximum for all plots, so they are all scaled to the same maximum
    # y_max = max(full_range)  # don't set a maximum 
    grid30=np.arange(0, 360, 30)  #  set a grid line every 30 degrees
    ax.set_thetagrids(grid30, labels=grid30,weight='bold')
    ax.set_rgrids(np.arange(0, y_max, int(y_max/4)), angle=0, weight= 'black')
    ax.tick_params(axis="x", labelsize=17)
    ax.grid(linewidth=1.5, zorder=0)
    ax.set_theta_zero_location('E') # zero starts to the right
    ax.set_ylim([0,y_max])
    ax.set_yticklabels([])


def build_rose_hist(tstep,bin_size):
    # angles =  [167.9052429229879, 62.300527191945, 145.00797980144134, 
    #     63.43494882292201, 161.56505117707798, 60.94539590092286, 
    #     15.255118703057775, 135.0, 63.8860873697093, 121.34521414171567, 
    #     101.30993247402021, 59.17233770013197, 132.70938995736145]
    # lengths = [15, 40, 43, 53, 12, 91, 11, 13, 105, 111, 20, 126, 15]
    bins = np.arange(-(bin_size/2), 360+bin_size+1, bin_size)  # (-5, 366, 10) (-7.5, 368, 15)


    json_name = "segments_angles.json"
    with open(json_name, 'r') as file:
        data = json.load(file)

    # Initialize a variable to store the rose_histogram values
    rose_histogram = None
    angles = []
    full_range = []

    # Iterate through the data to find the matching timestep
    for entry in data:
        print(f'this_tstep {this_tstep}')
        if entry['timestep_n'] == float(tstep):
            angles = entry['angles']
            rose_histogram = entry['rose_histogram']
            lengths = entry['lengths']
            break
    if len(angles)>0:
        angles_in_bins, bins = np.histogram(angles, bins,weights=lengths)

        # Sum the last value with the first value.
        angles_in_bins[0] += angles_in_bins[-1]

        # shouldn't be necessary, but in case there are angles > 180, sum them to their corresponding 
        # angle between 0 and 180. This way the result is symmetric: bottom half is the same 
        # as top half but mirrored
        single_half = np.sum(np.split(angles_in_bins[:-1], 2), 0)
        full_range = np.concatenate([single_half,single_half])  # repeat the sequence twice
        print(f'full_range \n{full_range}')
        print(f'original rose_histogram \n{rose_histogram}')

    return full_range


def draw_rose(full_range,bin_size):
        # make plot
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='polar')
    plot_options(ax,0)
    ax.bar(np.deg2rad(np.arange(0, 360, bin_size)), full_range,
        width=np.deg2rad(bin_size), bottom=0.0, color=(0.2, 0.2, 0.2),
        edgecolor='k', linewidth=2,zorder=2)    # 10 -> 15
    return ax

bin_sizes = [10,15]
tsteps = ['022000','029000','033000','050000','067000','100000']
# tsteps = ['050000']
for bin_size in bin_sizes:
    for tstep in tsteps:
    #  = tsteps[0]
        full_range = build_rose_hist(tstep,bin_size)
        if len(full_range)>0:
            ax = draw_rose(full_range,bin_size)
            plt.savefig("rose_bin"+str(bin_size)+"_t"+str(tstep),dpi=200)
            plt.close()
# plt.show()