import numpy as np
import matplotlib.pyplot as plt
import json
import glob

def plot_options(ax,max_y_from_file):
    if max_y_from_file == 0:
        y_max = 500  # arbitrary number, the histograms are empty anyway
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

def draw_rose_plot(full_range: np.ndarray,fast_or_slow: str,tstep: str):
    # make plot
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='polar')
    plot_options(ax)
    ax.bar(np.deg2rad(np.arange(0, 360, 10)), full_range,
        width=np.deg2rad(10), bottom=0.0, color=(0.2, 0.2, 0.2),
        edgecolor='k', linewidth=2,zorder=2)    # 10 -> 15

    # save the maximum value in an external file
    max_file_name = "max_"+fast_or_slow+"_"+str(tstep)+".txt"
    f = open(max_file_name, "w")
    f.write(str(max(full_range)))
    f.close()
    return ax

def draw_rose_plot_double(double_hist: list,tstep):
    fast_hist = double_hist[0]
    slow_hist = double_hist[1]
    if len(fast_hist)>0:
        max_fast = max(fast_hist)
    else:
        max_fast = 0  # default value

    if len(slow_hist)>0:
        max_slow = max(slow_hist)
    else:
        max_slow = 0  # default value

    
    # max_y_from_file = max_dict.get(int(tstep), 'NaN')
    max_y_from_file = 'NaN'
    # print(f'tstep max {max_y_from_file}')
    if max_y_from_file == 'NaN':
        # if it doesn't have an entry in the dictionary from the file, take the maximum in the current histograms
        max_y_from_file = max(max_fast,max_slow)


    # make plot
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='polar')
    plot_options(ax,max_y_from_file)

    if len(fast_hist)>0:
        ax.bar(np.deg2rad(np.arange(0, 360, 10)), fast_hist,
            width=np.deg2rad(10), bottom=0.0, color=(1.0, 0.584, 0.0),alpha=0.8,
            edgecolor='k', 
            linewidth=1,zorder=2)    # 10 -> 15
        # max_fast = max(double_hist[0])
    
    if len(slow_hist)>0:
        ax.bar(np.deg2rad(np.arange(0, 360, 10)), slow_hist,
            width=np.deg2rad(10), bottom=0.0, color=(0.431, 0.416, 0.396),alpha=0.5,
            # hatch="xx",
            edgecolor='k', 
            linewidth=1,
            zorder=2)
        # max_slow = max(double_hist[1])

    max_of_both = max(max_fast,max_slow)  # take the max between the two max 
    max_file_name = "max_double_"+str(tstep)+".txt"
    f = open(max_file_name, "w")
    f.write(str(max_of_both))
    f.close()
    
    return ax

def get_rose_histogram(tstep,l):
    # Load the JSON data from the file
    json_name = "segments_angles_"+str(l)+".json"
    with open(json_name, 'r') as file:
        data = json.load(file)

    # Initialize a variable to store the rose_histogram values
    rose_histogram = None

    # Iterate through the data to find the matching timestep
    for entry in data:
        if entry['timestep_n'] == float(tstep):
            rose_histogram = entry['rose_histogram']
            break
    
    return rose_histogram


rose_histogram = []

def extract_number(filenames):
    tsteps = [i.split("_py_bb_")[1].split('_nx.png')[0] for i in filenames]
    return tsteps

# get all the time steps where there are fractures (=there are rose diagrams)
p_files1 = sorted(glob.glob("rose_weight_p_1_py_bb_*_nx.png"))
tsteps1 = extract_number(p_files1)
p_files2 = sorted(glob.glob("rose_weight_p_2_py_bb_*_nx.png"))
tsteps2 = extract_number(p_files2)
p_files3 = sorted(glob.glob("rose_weight_p_3_py_bb_*_nx.png"))
tsteps3 = extract_number(p_files3)
p_files4 = sorted(glob.glob("rose_weight_p_4_py_bb_*_nx.png"))
tsteps4 = extract_number(p_files4)
p_files5 = sorted(glob.glob("rose_weight_p_5_py_bb_*_nx.png"))
tsteps5 = extract_number(p_files5)
p_files6 = sorted(glob.glob("rose_weight_p_6_py_bb_*_nx.png"))
tsteps6 = extract_number(p_files6)
p_files7 = sorted(glob.glob("rose_weight_p_7_py_bb_*_nx.png"))
tsteps7 = extract_number(p_files7)

# get all the unique values of these time steps
tsteps = list(set(tsteps1 + tsteps2 + tsteps3 + tsteps4 + tsteps5 + tsteps6 + tsteps7))
tsteps.sort()

# read the maximum values associated with the timesteps
max_dict = {}
with open("../../../../max_values.txt", "r") as file:
    for line in file:
        line = line.strip()
        if not line:  # Skip if the line is empty
            continue
        key, value = map(int, line.split())  # split between timestep and max value
        # Store in dictionary
        max_dict[key] = value

# print(f'max_dict {max_dict}')

# tsteps = ['015000']

for tstep in tsteps:
    rose_histogram_double = []
    print(f'tstep: {tstep}')
    for fast_or_slow in ["fast","slow"]:
        if fast_or_slow == "fast":
            layers_n = ["2","4","6"]
        elif fast_or_slow == "slow":
            layers_n = ["1","3","5","7"]

        for l in layers_n:
            rose_histogram_new = get_rose_histogram(tstep,l)
            if rose_histogram_new is not None and any(value != 0 for value in rose_histogram_new):  # check if any are not zero
                if len(rose_histogram) > 0:
                    # sum the contribution from the new layer to the old histogram
                    # it combines all histograms into one for each layer type
                    rose_histogram = [x + y for x, y in zip(rose_histogram, rose_histogram_new)]
                    # print(f'rose_histogram\n {rose_histogram}')
                else:
                    rose_histogram = rose_histogram_new

        # ax = draw_rose_plot(rose_histogram,fast_or_slow,tstep)
        # plt.show()
        # plt.savefig("rose_"+fast_or_slow+"_t"+str(tstep),dpi=200)
        rose_histogram_double.append(rose_histogram)
        rose_histogram = []
    
    #  draw the combined rose diagrams


    ax = draw_rose_plot_double(rose_histogram_double,tstep)
    print(f'rose_histogram_double \n{rose_histogram_double}')
    plt.savefig("rose_double_t"+str(tstep),dpi=200)
    plt.clf()   # close figure
    plt.close()

    # plt.show()

"""
Find the highest of the max:

> find . \( -path "*/lr41/*" -o -path "*/lr42/*" -o -path "*/lr43/*" -o -path "*/lr44/*" -o -path "*/lr45/*" \) -name "max_double_014000.txt" -exec awk '{ if ($1 > max) max=$1 } END { print max }' {} +

 find . \( -path "*/lr43/*" -o -path "*/lr46/*" -o -path "*/lr47/*" \) -name "max_double_011000.txt" -exec awk '{ if ($1 > max) max=$1 } END { print max }' {} +
"""