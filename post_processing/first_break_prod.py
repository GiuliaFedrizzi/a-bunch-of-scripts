"""
Get the time of the first broken bond, version for 'prod' directories 

"""
import os as os
import glob
import matplotlib.pyplot as plt


from useful_functions import getFirstBrokenBond

dir_list = ['p26','p27']
data = {}

for p_dir in dir_list:
    data[p_dir] = {}
    os.chdir(p_dir)
    for visc_dir in sorted(glob.glob("visc*")):  # viscosity directory
        os.chdir(visc_dir)
        data[p_dir][visc_dir] = {}
        for mr_dir in sorted(glob.glob("vis*")):  # melt rate directory
            os.chdir(mr_dir)
            print(os.getcwd())
            try:
                time_of_first_bb = getFirstBrokenBond("latte.log")
            except:
                time_of_first_bb = float('NaN')
                #print("couldn't open the file.")

            # Store the number in the dictionary
            mr = mr_dir.split('_')[-1]  # This gets the '01', '05', or '09' part of the folder name
            data[p_dir][visc_dir][mr] = time_of_first_bb

            os.chdir('..')
        os.chdir('..')
    os.chdir('..')

# print(f'data: {data}')

# Create a plot with all the data, different colors for each 'p' directory and different line styles for each 'visc' directory

# Define markers and colors for better visualization
markers = ['o', 's', '^', 'D', 'x', '*']  # Different markers for each 'visc'
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Different colors for each 'p'

# Define line styles for each 'visc' directory
line_styles = ['-', '--', ':', '-.']

# Prepare the plot
plt.figure(figsize=(14, 7))
plt.title('Combined Plot for p and visc Directories')
plt.xlabel('Melt rate')
plt.ylabel('Time of first broken bond')
# plt.xticks([1, 5, 9], labels=['01', '05', '09'])
plt.grid(True)

# Generate plots
for p_index, p_dir in enumerate(data):
    for visc_index, visc_dir in enumerate(data[p_dir]):
        x_values = [1, 5, 9]
        y_values = [
            float(data[p_dir][visc_dir][str(x).zfill(2)]) if data[p_dir][visc_dir][str(x).zfill(2)] else None 
            for x in x_values
        ]
        plt.plot(
            x_values, y_values, 
            marker=markers[visc_index % len(markers)], 
            color=colors[p_index % len(colors)], 
            linestyle=line_styles[visc_index % len(line_styles)], 
            label=f'{p_dir} - {visc_dir}'
        )

# Show the legend
plt.legend()
plt.show()
