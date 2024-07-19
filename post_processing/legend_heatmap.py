import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math

# Generate x and y values from 1 to 9
# x = np.arange(1, 10)
# y = np.arange(1, 10)
x = [2,4,6,8]
y = [2,4,6,8]

# Create a grid of ratios y/x
ratio_grid = np.zeros((len(y), len(x)))

for i, y_val in enumerate(y):
    for j, x_val in enumerate(x):
        ratio_grid[i, j] = np.log10(x_val/y_val)


# List of all colormaps
colormaps = ['twilight_shifted','BrBG', 
    'PRGn', 'PiYG', 'PuOr_r', 'RdBu_r', 'RdGy', 
    'RdYlBu', 'RdYlGn', 
    'coolwarm']
print(f'len: {len(colormaps)},colormaps {colormaps}')

# Create a figure
ncols = int(math.ceil(math.sqrt(len(colormaps))))
nrows = int(math.ceil(len(colormaps) / ncols))
print(f'nrows {nrows}, ncols {ncols}')
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 15))
axes = axes.flatten()

# Plotting the heatmap
for ax, colmap in zip(axes, colormaps):
    cax = ax.imshow(ratio_grid,vmin=-1,vmax=1, cmap=colmap, origin='lower', extent=[min(x)-1, max(x)+1, min(y)-1, max(y)+1])
    # plt.colorbar(label='Ratio (y/x)')
    ax.set_xticks(x)
    ax.set_yticks(y)
    ax.set_title(colmap)
    # fig.colorbar(cax, ax=ax, label='Ratio (y/x)')

    # Adding text annotations
    for i in range(len(y)):
        for j in range(len(x)):
            ax.text(x[j], y[i], f'{10**(ratio_grid[i, j]):.2f}', ha='center', va='center')#, color='white',weight='bold')

# plt.tight_layout()
plt.show()
