import numpy as np
import matplotlib.pyplot as plt

# Generate x and y values from 1 to 9
x = np.arange(1, 10)
y = np.arange(1, 10)

# Create a grid of ratios y/x
ratio_grid = np.zeros((len(y), len(x)))

for i, y_val in enumerate(y):
    for j, x_val in enumerate(x):
        ratio_grid[i, j] = y_val / x_val

# Plotting the heatmap
plt.figure(figsize=(8, 8))
plt.imshow(ratio_grid, cmap='viridis', origin='lower', extent=[0.5, 9.5, 0.5, 9.5])
plt.colorbar(label='Ratio (y/x)')
plt.xticks(x)
plt.yticks(y)
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Heatmap of Ratios y/x')

# Adding text annotations
for i in range(len(y)):
    for j in range(len(x)):
        plt.text(j + 1, i + 1, f'{ratio_grid[i, j]:.4f}', ha='center', va='center', color='white')

plt.show()
