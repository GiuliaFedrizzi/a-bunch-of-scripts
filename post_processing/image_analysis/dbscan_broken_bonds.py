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

min_samples = 10

# read file without using pandas
# keep only x and y coordinates that have at least one broken bond
def read_and_filter_data_from_csv(file_path):
    all_data = pd.read_csv(file_path)
    bb_df = all_data[['x coord','y coord','Broken Bonds']]

    return bb_df


def dbscan_and_plot(X,e,min_samples):
    dbscan = DBSCAN(eps=e, min_samples=min_samples).fit(X)
    X['cluster'] = dbscan.labels_
    X=X[X['cluster']>-1]

    # get the number of clusters to create a custom colormap
    # num_clusters = len(unique_labels)
    # colours = plt.cm.hsv(np.linspace(0, 1, num_clusters))
    # custom_colormap = ListedColormap(colours)
    
    if len(X)>0:
        # X['edge'] = X['cluster']%2
        # Plotting
        plt.figure(figsize=(10, 6))

        n_elements = 30 # minimum number of elements in a cluster to be kept
        cluster_counts = X['cluster'].value_counts()  # count the number of points in each cluster
        large_clusters = cluster_counts[cluster_counts >= n_elements].index
        print(set(X['cluster'].value_counts()))
        X_filtered = X[X['cluster'].isin(large_clusters)]
        unique_labels = set(X_filtered['cluster'])

        # scatter = plt.scatter(x_coords, y_coords, c=dbscan.labels_, marker='o', s=10, alpha=0.75,edgecolor='k',linewidth=edges)#, cmap=custom_colormap) #  edgecolor='k',
        # plt.colorbar(scatter, label='Cluster Label')
        # plt.colorbar(plt.cm.ScalarMappable(cmap=custom_colormap),label='Cluster Label')
        scatter = sns.scatterplot(data=X_filtered,x='x coord',y='y coord',hue='cluster',palette="Paired",marker='o',alpha=0.75,s=10,edgecolor='k',linewidth=0)
        sns.color_palette("hls", 20)

        # Calculate the centroids of each cluster or choose representative points
        centroids = X_filtered.groupby('cluster').mean().reset_index()

        # Loop through the centroids to annotate the cluster number
        for index, row in centroids.iterrows():
            plt.text(row['x coord'], row['y coord'], f"{row['cluster']}", horizontalalignment='center', size='medium', color='black', weight='semibold')

        plt.title('DBSCAN Clustering, eps ='+str(e)+', min_samples = '+str(min_samples) + ', min elements in a cluster = '+str(n_elements))
        plt.xlabel('x coord')
        plt.ylabel('y coord')

        plt.gca().set_aspect('equal')  # aspect ratio = image is a square
        # plt.colorbar(scatter, label='Cluster Label')

        scatter.legend(ncol=int(len(unique_labels)/10),loc='center left',bbox_to_anchor=(1.05, 0.5))

# plt.show()


data = read_and_filter_data_from_csv(filename)

# Preparing the data for DBSCAN
X = data[data['Broken Bonds'] > 0] # Only where there are bb 
X = X[['x coord','y coord']] # Only x and y coordinates 

for e in [0.0088,0.00995,0.01,0.012]:
# for e in [0.005, 0.008]:
    dbscan_and_plot(X,e,min_samples)

plt.show()  # show all of them at the end

if False:
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

