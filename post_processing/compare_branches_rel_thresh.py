import os
import pandas as pd
import matplotlib.pyplot as plt
from useful_functions import getParameterFromInput

def plot_data(root_dirs, visc, vis, row_index,custom_labels):
    # Create a figure for plotting
    fig, axs = plt.subplots(2, 3, figsize=(9, 6))
    axs = axs.flatten()

    # Define the metrics to plot
    metrics = ['n_I', 'n_2', 'n_3', 'n_4', 'n_5', 'branches_tot_length']
    print(f'visc: {visc}, vis: {vis}')
    label_mapping = dict(zip(root_dirs, custom_labels))
    # Loop over each metric and plot
    for i, metric in enumerate(metrics):
        for vis in vis_dirs:
            data = {}
            
            x = []
            y = []
            for root_dir in root_dirs:
                # define temporary row index (timestep)
                temp_row_index = row_index
                print(f'root_dir {root_dir}')
                # Construct the path to the csv file
                csv_file = os.path.join(root_dir, visc, vis, "py_branch_info.csv")
                print(f'csv_file {csv_file}')
                if os.path.exists(csv_file):
                    # Read the csv file
                    df = pd.read_csv(csv_file, sep=',')
                    # print(f'df: {df}')
                    if os.getcwd() == '/nobackup/scgf/myExperiments/threeAreas/prod/prt/prt01':
                        if root_dir == 'rt0.004' or root_dir == 'rt0.006' or root_dir == 'rt0.008':
                            temp_row_index *= 10
                            print(f'row_index {temp_row_index}')
                    if vis.endswith("02"):
                        temp_row_index = round(temp_row_index / 2)
                        print(f'tstep = {temp_row_index} (should be {row_index / 2})')
                    if vis.endswith("05"):
                        temp_row_index = round(temp_row_index / 5)
                        print(f'tstep = {temp_row_index} (should be {row_index / 5})')
                    if vis.endswith("08"):
                        temp_row_index = round(temp_row_index / 8)
                        print(f'tstep = {temp_row_index} (should be {row_index / 8})')
                    if temp_row_index < len(df):
                        x.append(label_mapping[root_dir])
                        y.append(df.loc[temp_row_index, metric])
                    else:
                        # Skip if row_index is out of range
                        print(f'Out of range')
                        continue
                else:
                    # Skip if the file is not found
                    print(f'-----file not found')
                    continue

            # Plot the data
            axs[i].plot(x, y, marker='o', label=vis)

        axs[i].set_title(metric)
        axs[i].set_xlabel('rel thresh')
        axs[i].set_ylabel(metric)
        axs[i].legend()
        axs[i].tick_params(axis='x', labelsize=5)

    # plt.tight_layout()
    fig.suptitle(visc+", t$_{ref}$ = "+str(row_index), fontsize=16)
    plt.show()


# root_dirs =     ['p52',    'p53',   'p50',    'p61',     'p62',    'p56',       'p63']#, 'p50']  # Add your p* directories here
# custom_labels = ['0.001', '0.0015','0.002','0.001, 2','0.001, 4','0.001, 10','0.001, 20']    # Custom labels
# root_dirs =     ['rt0.006',    'rt0.01',   'rt0.02','rt0.04']#,    'p61',     'p62',    'p56',       'p63']#, 'p50'] 
entries = os.listdir('.')
root_dirs = sorted([entry for entry in entries if entry.startswith('rt0') and os.path.isdir(entry)])
print(f'dirs are {root_dirs}')
visc = 'visc_4_1e4'#,'visc_4_1e4']
vis_dirs = ['vis1e4_mR_02', 'vis1e4_mR_05','vis1e4_mR_08']  # Add your selected vis* subdirectories here
row_index = 100  # Specify the row index = timestep


custom_labels = []
for d in root_dirs:
    custom_labels.append(getParameterFromInput(d+"/"+visc+"/baseFiles/input.txt","second_rel_thr"))   # extract the value for relaxation threshold
print(f'rel_thresh {custom_labels}')

# custom_labels = ['0.001', '0.0015','0.002']#,'0.001, 2','0.001, 4','0.001, 10','0.001, 20']    # Custom labels

if len(root_dirs) != len(custom_labels):
    custom_labels = root_dirs


# for visc in viscs:
plot_data(root_dirs, visc, vis_dirs, row_index, custom_labels)
