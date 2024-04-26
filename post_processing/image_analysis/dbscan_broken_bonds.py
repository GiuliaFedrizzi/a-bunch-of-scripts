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
import pandas as pd

filename = '/nobackup/scgf/myExperiments/threeAreas/prod/prt/prt45/rt0.5/visc_3_1e3/vis1e3_mR_08/my_experiment16000.csv'

min_samples = 15

# read file without using pandas
# keep only x and y coordinates that have at least one broken bond
def read_and_filter_data_from_csv(file_path):
    all_data = pd.read_csv(file_path)
    bb_df = all_data[['x coord','y coord','Broken Bonds']]

    # with open(file_path, newline='') as csvfile:
    #     reader = csv.DictReader(csvfile)
    #     for row in reader:
    #         x = float(row['x coord'])
    #         y = float(row['y coord'])
    #         broken_bonds = int(row['Broken Bonds'])
    #         if broken_bonds >= 1:
    #             all_data.append((x, y, broken_bonds))
    return bb_df


def dbscan_and_plot(X,e,min_samples):
    dbscan = DBSCAN(eps=e, min_samples=min_samples).fit(X)
    X['cluster'] = dbscan.labels_
    
    unique_labels = set(X['cluster'])

    # get the number of clusters to create a custom colormap
    num_clusters = len(unique_labels)
    colours = plt.cm.hsv(np.linspace(0, 1, num_clusters))
    custom_colormap = ListedColormap(colours)
    
    X['edge'] = X['cluster']%2
    
    # Plotting
    plt.figure(figsize=(10, 6))

    # scatter = plt.scatter(x_coords, y_coords, c=dbscan.labels_, marker='o', s=10, alpha=0.75,edgecolor='k',linewidth=edges)#, cmap=custom_colormap) #  edgecolor='k',
    # plt.colorbar(scatter, label='Cluster Label')
    # plt.colorbar(plt.cm.ScalarMappable(cmap=custom_colormap),label='Cluster Label')
    scatter = plt.scatter(X['x coord'],X['y coord'],c=X['cluster'],marker='o',alpha=0.75,s=10,edgecolor='k',linewidth=X['edge'])
    plt.title('DBSCAN Clustering, eps ='+str(e))
    plt.xlabel('x coord')
    plt.ylabel('y coord')

    plt.gca().set_aspect('equal')  # aspect ratio = image is a square
    plt.colorbar(scatter, label='Cluster Label')

# plt.show()


data = read_and_filter_data_from_csv(filename)

# Preparing the data for DBSCAN
X = data[data['Broken Bonds'] > 0] # Only where there are bb 
X = X[['x coord','y coord']] # Only x and y coordinates 

for e in [0.005,0.01,0.012,0.02,0.03]:
# for e in [0.005, 0.008]:
    dbscan_and_plot(X,e,min_samples)

# plt.show()  # show all of them at the end


# Use NearestNeighbors to find the distance to the k-th nearest neighbor
k = min_samples  # Same as your DBSCAN min_samples value
nearest_neighbors = NearestNeighbors(n_neighbors=k)
neighbors_fit = nearest_neighbors.fit(X)
distances, indices = neighbors_fit.kneighbors(X)

# Sort the distances
distances = np.sort(distances[:, k-1], axis=0)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(distances)
plt.xlabel('Points sorted by distance to the 5th nearest neighbor')
plt.ylabel(str(min_samples)+'th nearest neighbor distance')
plt.title(str(min_samples)+'th Nearest Neighbor Distance vs. Points')
plt.grid(True)
plt.show()

