# import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import csv
from scipy.stats import gaussian_kde
from scipy.spatial.distance import cdist
from matplotlib.colors import ListedColormap
from matplotlib.patches import Ellipse
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
import colorcet  as cc
import pandas as pd
import sys
sys.path.append('/home/home01/scgf/myscripts/post_processing')

from useful_functions import getSaveFreq

#filename = '/nobackup/scgf/myExperiments/threeAreas/prod/prt/prt45/rt0.5/visc_3_1e3/vis1e3_mR_08/my_experiment07000.csv'
#filename = '/nobackup/scgf/myExperiments/threeAreas/prod/prt/prt45/rt0.5/visc_3_1e3/vis1e3_mR_08/my_experiment16000.csv'
filename = '/nobackup/scgf/myExperiments/threeAreas/prod/prt/prt45/rt0.5/visc_3_1e3/vis1e3_mR_08/my_experiment12000.csv'
#filename = '/nobackup/scgf/myExperiments/threeAreas/prod/prt/prt45/rt0.5/visc_1_1e1/vis1e1_mR_06/my_experiment70000.csv'

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
    X=X[X['cluster']>-1]  # remove noise cluster
    
    if len(X)>0:
        # Plotting
        plt.figure(figsize=(10, 6))

        n_elements = 40 # minimum number of elements in a cluster to be kept
        cluster_counts = X['cluster'].value_counts()  # count the number of points in each cluster
        large_clusters = cluster_counts[cluster_counts >= n_elements].index
        X_filtered = X[X['cluster'].isin(large_clusters)]
        if len(X_filtered)>0:
            # define custom palette with as many colours as there are clusters
            unique_labels = set(X_filtered['cluster'])
            palette = sns.color_palette(cc.glasbey, n_colors=len(unique_labels))

            # scatter = plt.scatter(x_coords, y_coords, c=dbscan.labels_, marker='o', s=10, alpha=0.75,edgecolor='k',linewidth=edges)#, cmap=custom_colormap) #  edgecolor='k',
            scatter = sns.scatterplot(data=X_filtered,x='x coord',y='y coord',hue='cluster',palette=palette,marker='o',alpha=0.75,s=10,edgecolor='k',linewidth=0)
            
            plt.title('DBSCAN Clustering, eps ='+str(e)+', min_samples = '+str(min_samples) + ', min elements in a cluster = '+str(n_elements)+'\n'+filename)
            plt.xlabel('x coord')
            plt.ylabel('y coord')

            plt.gca().set_aspect('equal')  # aspect ratio = image is a square
            # plt.colorbar(scatter, label='Cluster Label')

            legend_columns = int(len(unique_labels)/20)
            if legend_columns < 2:
                legend_columns = 2
            scatter.legend(ncol=legend_columns,loc='center left',bbox_to_anchor=(1.05, 0.5))
            
            # Calculate the centroids of each cluster or choose representative points
            centroids = X_filtered.groupby('cluster').mean()#.reset_index()
            # Moments analysis
            moments_of_inertia = {}
            all_eigenvectors = {}
            all_elong = {}
            all_angles = {}
            all_sizes = {}
            for cluster_label, centroid in centroids.iterrows():
                # Select the points that belong to the current cluster
                cluster_points = X_filtered[X_filtered['cluster'] == cluster_label][['x coord', 'y coord']]

                # Subtract the centroid from the cluster points
                cluster_points -= centroid
                # Calculate the covariance matrix
                covariance_matrix = np.cov(cluster_points, rowvar=False)
                
                # Calculate the eigenvalues (moment of inertia components) and eigenvectors
                eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
                
                order = eigenvalues.argsort()[::-1] # sort in descending order
                eigenvalues = eigenvalues[order]
                eigenvectors = eigenvectors[:,order]
                largest_eigenvector = eigenvectors[:,0]  # take the largest
                angle = np.arctan2(largest_eigenvector[1], largest_eigenvector[0]) # arctan is the angle in radians
                angle_degrees = np.degrees(angle)
                theta = np.linspace(0, 2*np.pi, 1000);
                # draw an ellipse:
                eigenvalues_draw = eigenvalues*3; eigenvectors_draw = eigenvectors
                ellipsis = (np.sqrt(eigenvalues_draw[None,:]) * eigenvectors_draw) @ [np.sin(theta), np.cos(theta)] # parametric form
                plt.plot(ellipsis[0,:]+centroid['x coord'], ellipsis[1,:]+centroid['y coord'],'k')

                # Store the eigenvalues in the dictionary
                moments_of_inertia[cluster_label] = eigenvalues
                all_eigenvectors[cluster_label] = eigenvectors
                all_elong[cluster_label] = eigenvalues[0]/eigenvalues[1]
                all_angles[cluster_label] = angle_degrees
                all_sizes[cluster_label] = len(cluster_points)+1
            
            #compute some global data
            weighted_sum = sum(angle * all_sizes[cluster] for cluster, angle in all_angles.items())
            weighted_average_angle = weighted_sum / sum(all_sizes.values())
            weighted_elong = sum(elong * all_sizes[cluster] for cluster, elong in all_elong.items())
            weighted_average_elong = weighted_elong / sum(all_sizes.values())

            cluster_data = {'cluster_n':len(all_sizes),'average_size':sum(all_sizes.values())/len(all_sizes),'average_angle':weighted_average_angle,'average_elong':weighted_average_elong}
            print(cluster_data)
            
            #print(f'moments of inertia \n{moments_of_inertia}')
            # Loop through the centroids to annotate the cluster number
            for index, row in centroids.iterrows():
                plt.text(row['x coord'], row['y coord'], f"{int(index)}", horizontalalignment='center', size='medium', color='black', weight='semibold')
            
    else:
        print("Clusters are too small")
# plt.show()

df = pd.DataFrame()
data = read_and_filter_data_from_csv(filename)
X = data[data['Broken Bonds'] > 0] # Only where there are bb 
X = X[['x coord','y coord']] # Only x and y coordinates 
dbscan_and_plot(X,0.0088,min_samples)

#dbscan_and_plot(X,0.0088,min_samples)
#for e in [0.0088,0.00999, 0.01]:
#    # I need to reset what X is at every cycle
#    X = data[data['Broken Bonds'] > 0] # Only where there are bb 
#    X = X[['x coord','y coord']] # Only x and y coordinates 
#    dbscan_and_plot(X,e,min_samples)

plt.show()  # show all of them at the end

if False:
    X = data[data['Broken Bonds'] > 0] # Only where there are bb 
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

