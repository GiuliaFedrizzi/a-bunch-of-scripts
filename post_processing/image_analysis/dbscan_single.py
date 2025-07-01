"""
plot one figure, from one my_experiment file
to be run from outside rt0.5/

Giulia June 2024
"""
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from dbscan_functions import read_bb_data_from_csv,dbscan_and_plot,draw_rose_ellipse
import os
import sys
from sklearn.neighbors import NearestNeighbors

no_margins = 1
plot_figures = 1
t = 11000  # true time!
# t =25000 # true time!
# os.chdir("rt0.5/pdef0e8/")
os.chdir("rt0.5/visc_3_1e3/")  # change  x_value and melt_value too
file_to_read = "my_experiment"+str(t)+".csv"
x_value = "1e3"      # <----------
melt_value = 0.008    # <----------
data = read_bb_data_from_csv("vis1e3_mR_08/"+file_to_read)
# data = read_bb_data_from_csv("vis1e2_mR08/"+file_to_read)
print("vis1e3_mR_08/"+file_to_read)
if data.empty:
    print("empty df") 
    sys.exit()


X = data[data['Broken Bonds'] > 0] # Only where there are bb 

if no_margins:
    X = X[X['x coord'] > 0.02] # exclude the left margin
    X = X[X['y coord'] > 0.02] # exclude the bottom margin
    X = X[X['x coord'] < 0.98] # # exclude the right margin 
    X = X[X['y coord'] < 0.98] # # exclude the top margin 

X = X[['x coord','y coord']] # Only x and y coordinates 
if len(X) > 0:
    cluster_df,ellipses_df,_ = dbscan_and_plot(X,"viscosity",x_value,melt_value,no_margins,plot_figures,t)
    ellipses_df['average_angle_from_90'] = abs(90-ellipses_df['Angle_Degrees'])
    print(ellipses_df)
    if True:
        ax = draw_rose_ellipse(ellipses_df)
        plt.show()
        #  warning: t is the true time, not the reference time (used in dbscan_broken_bonds)
        # plt.savefig("../cluster/rose_"+str(x_value)+"_"+str(melt_value)+"_"+str(t)+".png",transparent=True)


if False:
    min_samples = 10
    # Use NearestNeighbors to find the distance to the k-th nearest neighbor
    k = min_samples  # Same as your DBSCAN min_samples value
    nearest_neighbors = NearestNeighbors(n_neighbors=k)
    neighbors_fit = nearest_neighbors.fit(X)
    distances, indices = neighbors_fit.kneighbors(X)

    # Sort the distances
    distances = np.sort(distances[:, k-1], axis=0)

    # Plotting
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['axes.linewidth'] = 1
    # plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['xtick.labelsize'] = 13
    plt.rcParams['ytick.labelsize'] = 13
    plt.rcParams['xtick.major.size'] = 3
    plt.rcParams['xtick.major.width'] = 1
    plt.rcParams['ytick.major.size'] = 3
    plt.rcParams['ytick.major.width'] = 1

    plt.figure(figsize=(10, 8))
    plt.plot(distances)
    plt.xlabel('Points sorted by distance to the '+str(min_samples)+'th nearest neighbour',fontsize=15,weight='bold')
    plt.ylabel("$\epsilon$",fontsize=15)
    # plt.xlim([0,25000])
    plt.ylim([0.008,0.02])
    # plt.axhline(y=0.0088,color='silver',linestyle='--')
    plt.axhline(y=0.00995,color='silver',linestyle='--')  # line to show where the chosen epsilon is
    # plt.title(str(min_samples)+'th Nearest Neighbor Distance vs. Points')
    plt.grid(True,which='both')
    plt.show()
    # plt.savefig("../cluster/dbscan_kNN_epsion.png")


# print(cluster_df)
print("the end")

