"""
plot one figure, from one my_experiment file
to be run from outside rt0.5/

Giulia June 2024
"""
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from dbscan_functions import read_bb_data_from_csv,dbscan_and_plot
import os
import sys
from sklearn.neighbors import NearestNeighbors

no_margins = 1
plot_figures = 1
t = 16000
os.chdir("rt0.5/visc_3_1e3/")
file_to_read = "my_experiment"+str(t)+".csv"
x_value = 1e3
melt_value = 0.008
data = read_bb_data_from_csv("vis1e3_mR_08/"+file_to_read)

if data.empty:
    print("empty df") 
    sys.exit()


X = data[data['Broken Bonds'] > 0] # Only where there are bb 

if no_margins:
    X = X[X['x coord'] > 0.02] # exclude the left margin
    X = X[X['x coord'] < 0.98] # # exclude the right margin 
X = X[['x coord','y coord']] # Only x and y coordinates 
# if len(X) > 0:
#     cluster_df = dbscan_and_plot(X,"viscosity",x_value,melt_value,no_margins,plot_figures,t)
print(f'len X {len(X)}')

min_samples = 10
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
plt.xlabel('Points sorted by distance to the '+str(min_samples)+'th nearest neighbor')
plt.ylabel("$\epsilon$")
plt.xlim([0,25000])
plt.ylim([0.008,0.02])
plt.axhline(y=0.0088,color='silver',linestyle='--')
# plt.title(str(min_samples)+'th Nearest Neighbor Distance vs. Points')
plt.grid(True)
# plt.show()
plt.savefig("../cluster/dbscan_kNN_epsion.png")


# print(cluster_df)
print("the end")

