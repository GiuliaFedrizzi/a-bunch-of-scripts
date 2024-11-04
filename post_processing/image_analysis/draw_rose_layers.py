import numpy as np
import matplotlib.pyplot as plt
import json
import glob

def draw_rose_plot(full_range: np.ndarray,fast_or_slow: str):
    # make plot
    print(f'full_range {full_range}')
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='polar')
    # y_max = 535  # set a maximum for all plots, so they are all scaled to the same maximum
    y_max = max(full_range)  # don't set a maximum 
    grid30=np.arange(0, 360, 30)  #  set a grid line every 30 degrees
    ax.set_thetagrids(grid30, labels=grid30,weight='bold')
    ax.set_rgrids(np.arange(0, y_max, 425), angle=0, weight= 'black')
    ax.tick_params(axis="x", labelsize=17)
    ax.grid(linewidth=1.5, zorder=0)
    ax.set_theta_zero_location('E') # zero starts to the right
    ax.bar(np.deg2rad(np.arange(0, 360, 10)), full_range,
        width=np.deg2rad(10), bottom=0.0, color=(0.2, 0.2, 0.2),
        edgecolor='k', linewidth=2,zorder=2)    # 10 -> 15
        # width=np.deg2rad(10), bottom=0.0, color=(1, 0, 0), edgecolor='r')
    ax.set_ylim([0,y_max])
    ax.set_yticklabels([])
    max_file_name = "max_"+fast_or_slow+".txt"
    f = open(max_file_name, "w")
    f.write(str(max(full_range)))
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

tsteps = list(set(tsteps1 + tsteps2 + tsteps3 + tsteps4 + tsteps5 + tsteps6 + tsteps7))
tsteps.sort()

print(f'tsteps {tsteps}')
for tstep in tsteps:
    for fast_or_slow in ["fast","slow"]:
        if fast_or_slow == "fast":
            layers_n = ["2","4","6"]
        elif fast_or_slow == "slow":
            layers_n = ["1","3","5","7"]

        for l in layers_n:
            # tstep = 9000
            rose_histogram_new = get_rose_histogram(tstep,l)
            if rose_histogram_new is not None and any(value != 0 for value in rose_histogram_new):  # check if any are not zero
                print(f'\nlayer {l}')
                print(f'rose_histogram_new\n {rose_histogram_new}')
                if len(rose_histogram) > 0:
                    rose_histogram = [x + y for x, y in zip(rose_histogram, rose_histogram_new)]
                    print(f'rose_histogram\n {rose_histogram}')
                else:
                    rose_histogram = rose_histogram_new



        ax = draw_rose_plot(rose_histogram,fast_or_slow)
        # plt.show()
        plt.savefig("rose_"+fast_or_slow+"_t"+str(tstep),dpi=200)
        rose_histogram = []
