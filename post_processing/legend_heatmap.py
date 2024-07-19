import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
from matplotlib.ticker import FuncFormatter

# Generate x and y values from 1 to 9
x = [0,2,4,6,8]
y = [2,4,6,8]

# Create a grid of ratios y/x
ratio_grid = np.zeros((len(y), len(x)))

for i, y_val in enumerate(y):
    for j, x_val in enumerate(x):
        ratio_grid[i,j] = np.log10(x_val/y_val)

print(f'\n{ratio_grid}')

colmap = 'RdBu_r'

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))

cax = ax.imshow(ratio_grid,vmin=-1,vmax=1, cmap=colmap, origin='lower', extent=[min(x)-1, max(x)+1, min(y)-1, max(y)+1])
# plt.colorbar(label='Ratio (y/x)')
ax.set_xticks(x,[f'{i*1e-8:.0e}' for i in x])
ax.set_yticks(y, [j/1000 for j in y])
# fig.colorbar(cax, ax=ax, label='Ratio')

# Adding text annotations
for i in range(len(y)):
    for j in range(len(x)):
        ax.text(x[j], y[i], f'{10**(ratio_grid[i, j]):.2f}', ha='center', va='center')#, color='white',weight='bold')

# plt.tight_layout()
plt.savefig("legend_def_mrate.png",dpi=300)
plt.show()
