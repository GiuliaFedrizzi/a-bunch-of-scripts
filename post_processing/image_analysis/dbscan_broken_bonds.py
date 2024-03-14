# import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import csv
from scipy.stats import gaussian_kde
from matplotlib.colors import ListedColormap
from scipy.spatial.distance import pdist, squareform
import seaborn as sns

filename = '/nobackup/scgf/myExperiments/threeAreas/prod/prt/prt34/rt0.5/visc_1_1e1/vis1e1_mR_08/my_experiment10000.csv'
# myExp = pd.read_csv(filename, header=0)

# read file without using pandas
# keep only x and y coordinates that have at least one broken bond
def read_and_filter_data_from_csv(file_path):
    all_data = []
    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            x = float(row['x coord'])
            y = float(row['y coord'])
            broken_bonds = int(row['Broken Bonds'])
            if broken_bonds >= 1:
                all_data.append((x, y, broken_bonds))
    return np.array(all_data)


def dbscan_and_plot(X,e):
    dbscan = DBSCAN(eps=e, min_samples=5).fit(X)

    # myExp['cluster'] = dbscan.labels_

    # get the number of clusters to create a custom colormap
    num_clusters = len(np.unique(dbscan.labels_))
    colors = plt.cm.hsv(np.linspace(0, 1, num_clusters))
    
    custom_colormap = ListedColormap(colors)
    
    edges = [i%2 for i in dbscan.labels_]
    # print(f'edges {edges}')
    # Plotting
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(x_coords, y_coords, c=dbscan.labels_, marker='o', s=10, alpha=0.75,edgecolor='k',linewidth=edges, cmap=custom_colormap) #  edgecolor='k',
    plt.colorbar(scatter, label='Cluster Label')
    plt.title('DBSCAN Clustering, eps ='+str(e))
    plt.xlabel('x coord')
    plt.ylabel('y coord')

    # Highlight noise points
    noise_points = dbscan.labels_ == -1
    plt.scatter(x_coords[noise_points], y_coords[noise_points], color='k', marker='x', s=10, label='Noise')
    plt.gca().set_aspect('equal')  # aspect ratio = image is a square
    plt.legend()

# plt.show()


data = read_and_filter_data_from_csv(filename)
x_coords, y_coords, broken_bonds = data[:,0], data[:,1], data[:,2]

# Preparing the data for DBSCAN
X = data[:, :2]  # Only x and y coordinates

for e in [0.015, 0.02, 0.025]:
    dbscan_and_plot(X,e)
# X = myExp[["x coord", "y coord"]]

plt.show()  # show all of them at the end


# # Use NearestNeighbors to find the distance to the k-th nearest neighbor
# k = 5  # Same as your DBSCAN min_samples value
# nearest_neighbors = NearestNeighbors(n_neighbors=k)
# neighbors_fit = nearest_neighbors.fit(X)
# distances, indices = neighbors_fit.kneighbors(X)

# # Sort the distances
# distances = np.sort(distances[:, k-1], axis=0)

# # Plotting
# plt.figure(figsize=(10, 6))
# plt.plot(distances)
# plt.xlabel('Points sorted by distance to the 5th nearest neighbor')
# plt.ylabel('5th nearest neighbor distance')
# plt.title('5th Nearest Neighbor Distance vs. Points')
# plt.grid(True)
# plt.show()

