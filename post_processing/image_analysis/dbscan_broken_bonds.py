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
import os
import matplotlib.gridspec as gridspec

sys.path.append('/home/home01/scgf/myscripts/post_processing')

from useful_functions import getSaveFreq,getParameterFromLatte
from viz_functions import find_dirs,find_variab

save_freq = int(getSaveFreq())
no_margins = 0

# times = list(range(1, 20, 1)) + list(range(20, 141, 5)) + list(range(150, 501, 20)) + list(range(500, 801, 20)) + list(range(850, 1500, 40))
times = [100]
times = [i*save_freq for i in times]   # convert to timestep


min_samples = 10
plot_figures = False  # if I want to visualise clusters, ellipses etc

# read file without using pandas
# keep only x and y coordinates that have at least one broken bond
def read_and_filter_data_from_csv(file_path):
    # print(file_path)
    if os.path.isfile(file_path):
        all_data = pd.read_csv(file_path)
        bb_df = all_data[['x coord','y coord','Broken Bonds']]

        return bb_df
    else:
        return pd.DataFrame() 

def dbscan_and_plot(X,e,min_samples,df,variab,x_value,melt_value,no_margins):
    dbscan = DBSCAN(eps=e, min_samples=min_samples).fit(X)
    X['cluster'] = dbscan.labels_
    X=X[X['cluster']>-1]  # remove noise cluster

    if len(X)>0:
        n_elements = 40 # minimum number of elements in a cluster to be kept
        cluster_counts = X['cluster'].value_counts()  # count the number of points in each cluster
        large_clusters = cluster_counts[cluster_counts >= n_elements].index
        X_filtered = X[X['cluster'].isin(large_clusters)]
        # Calculate the centroids of each cluster or choose representative points
        centroids = X_filtered.groupby('cluster').mean()#.reset_index()

        if len(X_filtered)>0:
            if plot_figures:
                plt.figure(figsize=(10, 6))
                # define custom palette with as many colours as there are clusters
                unique_labels = set(X_filtered['cluster'])
                palette = sns.color_palette(cc.glasbey, n_colors=len(unique_labels))

                # scatter = plt.scatter(x_coords, y_coords, c=dbscan.labels_, marker='o', s=10, alpha=0.75,edgecolor='k',linewidth=edges)#, cmap=custom_colormap) #  edgecolor='k',
                scatter = sns.scatterplot(data=X_filtered,x='x coord',y='y coord',hue='cluster',palette=palette,marker='o',alpha=0.75,s=10,edgecolor='k',linewidth=0)
                
                plt.title('DBSCAN Clustering, eps ='+str(e)+', min_samples = '+str(min_samples) + ', min elements in a cluster = '+str(n_elements)+'\n'+os.getcwd())
                plt.xlabel('x coord')
                plt.ylabel('y coord')
                plt.xlim([0,1])
                plt.ylim([0,1])
                plt.gca().set_aspect('equal')  # aspect ratio = image is a square
                # plt.colorbar(scatter, label='Cluster Label')

                legend_columns = int(len(unique_labels)/20)
                if legend_columns < 2:
                    legend_columns = 2
                scatter.legend(ncol=legend_columns,loc='center left',bbox_to_anchor=(1.05, 0.5))
                
                # Loop through the centroids to annotate the cluster number
                for index, row in centroids.iterrows():
                    plt.text(row['x coord'], row['y coord'], f"{int(index)}", horizontalalignment='center', size='medium', color='black', weight='semibold')

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
                if plot_figures:
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

            cluster_data = {'cluster_n':len(all_sizes),'average_size':sum(all_sizes.values())/len(all_sizes),'average_angle':weighted_average_angle,
            'average_angle_from_90':abs(90-weighted_average_angle),'average_elong':weighted_average_elong,variab:x_value,'melt_rate':melt_value}
            
        else:
            """ clusters were found, but they are too small: populate the dataframe with NaN """
            cluster_data = {'cluster_n':float('NaN'),'average_size':float('NaN'),'average_angle':float('NaN'),
            'average_angle_from_90':float('NaN'),'average_elong':float('NaN'),variab:x_value,'melt_rate':melt_value}
            print("Clusters are too small")

    else:
        """ no clusters were found: populate the dataframe with NaN """
        cluster_data = {'cluster_n':float('NaN'),'average_size':float('NaN'),'average_angle':float('NaN'),
            'average_angle_from_90':float('NaN'),'average_elong':float('NaN'),variab:x_value,'melt_rate':melt_value}
        print("No Clusters")
        
    # add the new data to the existing dataframe
    cluster_df = pd.DataFrame(cluster_data,index=[0])
    df = pd.concat([df,cluster_df], ignore_index=True) 
    
    return df

def plot_cluster_data(fig,df,cluster_variable,cluster_variable_label,no_margins):
    # set which figure to use
    plt.figure(fig)
    plt.clf()

    # prepare to handle NaNs
    mask_nan = df[cluster_variable].isna()

    # prepare to highlight big values
    smallest_value = df[cluster_variable].min()
    big_value = (df[cluster_variable].max() - smallest_value) * 3/4 + smallest_value
    print(f'{cluster_variable_label}: big_value {big_value}, max {df[cluster_variable].max()}, min {df[cluster_variable].min()}')
    df_big_values_mask = df[cluster_variable] > big_value


    gs = gridspec.GridSpec(2, 2, width_ratios=[5, 2], height_ratios=[2, 7])
    # Adjust layout to prevent overlap
    # plt.tight_layout()
    fig.suptitle(cluster_variable_label+", t = " + str(t))
    unique_viscosity_values = set(df['viscosity'])
    #prepare labels
    viscosity_labels = {}
    for v in unique_viscosity_values:
        viscosity_labels[v] = "$10^{"+str(np.log10(v))+"}$"
    df['viscosity_label'] = df['viscosity'].map(viscosity_labels)
    df['melt_rate'] = df['melt_rate'].astype(str)

    # pseudo heatmap
    square_size = 2800
    ax0 = plt.subplot(gs[1, 0])
    # sns.heatmap(heatmap_data, ax=ax0, cmap="viridis", cbar_kws={'label': 'Average Size'})
    sns.scatterplot(data=df, x='viscosity',y='melt_rate',hue=cluster_variable,
                    marker='s',s=square_size,edgecolor='none', ax=ax0, palette="crest")
    ax0.set_xlabel("Viscosity")
    ax0.set_ylabel("Melt Rate")
    ax0.set_xscale('log')

    # Scatter plot for NaN values 
    sns.scatterplot(data=df[mask_nan], x='viscosity', y='melt_rate', color='gray',
                    marker='s', s=square_size,edgecolor='none', ax=ax0)

    # # highlight big values
    # sns.scatterplot(data=df[df_big_values_mask], x='viscosity', y='melt_rate',facecolors='none', hatch='///',
    #                 marker='s',edgecolor='black', s=square_size*0.6, ax=ax0)

    # Top plot (Viscosity vs Average Size)
    ax1 = plt.subplot(gs[0, 0], sharex=ax0)   # sharex  so they are aligned
    sns.lineplot(data=df, x="viscosity", y=cluster_variable,hue="melt_rate", ax=ax1, palette="viridis")
    ax1.set_xlabel("")
    ax1.set_ylabel(cluster_variable_label)
    ax1.legend(title="Melt Rate",ncol=1, fontsize=7, title_fontsize=9)
    ax1.xaxis.tick_top()

    # Right plot (Melt Rate vs Average Size)
    ax2 = plt.subplot(gs[1, 1], sharey=ax0)    # sharey  so they are aligned
    sns.lineplot(data=df, x=cluster_variable, y="melt_rate",hue="viscosity_label", ax=ax2, palette="viridis")
    ax2.set_ylabel("")
    # ax2.set_yticks("")  # no ticks - it removes them from the heatmap too
    ax2.set_xlabel(cluster_variable_label)
    ax2.legend(title="$\mu_f$", fontsize=7, title_fontsize=9)
    ax2.yaxis.tick_right()

    # Show or save the plot
    # plt.show()
    
    if no_margins:  # excluding margins, use a different name
        plt.savefig("cluster_x_"+cluster_variable+"_"+str(t)+".png")
    else:
        plt.savefig("cluster_"+cluster_variable+"_"+str(t)+".png")



def plot_pair_grid(df,t,no_margins):
    """
    make a grid with plots for each variable pair.
    Use melt rate and viscosity as hue (one figure for each hue)
    """

    fig_pair = plt.figure(figsize=(7, 7))
    # plt.title("t = "+str(t))

    df_pair = df.drop(columns=['average_angle'])   # keep only the columns that are useful for the grid plot
    df_pair = df_pair.dropna()

    # set which figure to use
    plt.figure(fig_pair)

    # hue is viscosity
    grid_mu = sns.PairGrid(df_pair.drop(columns=['melt_rate']), hue='viscosity', palette="crest") # (df.dropna())
    grid_mu = grid_mu.map_offdiag(sns.scatterplot)
    grid_mu = grid_mu.map_diag(sns.histplot)
    grid_mu.add_legend(title="Viscosity")
    
    #  if excluding margins:
    if no_margins:
        plt.savefig("cluster_x_grid_visc"+str(t)+".png")
    else:
        plt.savefig("cluster_grid_visc"+str(t)+".png")
    plt.clf()  #  clear the figure, ready to be reused

    # plt.title("t = "+str(t))

    # hue is melt rate
    grid_mr = sns.PairGrid(df_pair.drop(columns=['viscosity']), hue='melt_rate', palette="crest_r") # (df.dropna())
    grid_mr = grid_mr.map_offdiag(sns.scatterplot)
    grid_mr = grid_mr.map_diag(sns.histplot)
    grid_mr.add_legend(title="Melt Rate")
    
    #   if excluding margins:
    if no_margins:
        plt.savefig("cluster_x_grid_mr"+str(t)+".png")
    else:
        plt.savefig("cluster_grid_mr"+str(t)+".png")
    
    plt.close(fig_pair)  # we're done with the figures



def cluster_analysis(t,no_margins):
    """
    executed once for each t in time list
    reads the cluster dataframe from a file or creates one
    plots cluster analysis data
    """
    df = pd.DataFrame()

    #   if excluding margins:
    if no_margins:
        csv_name = 'cluster_data_x_'+str(t)+'.csv'
    else:
        csv_name = 'cluster_data_'+str(t)+'.csv'

    # check if there is already a file with the data. If there isn't, read and calculate
    if not os.path.isfile(csv_name):
        variab = find_variab()
        x_variable = find_dirs()  # viscosity dirs

        for x in x_variable:
            print(x)
            # if x == 'visc_2_1e2':
            if variab == "viscosity":
                x_value = float(getParameterFromLatte(x+'/baseFiles/input.txt','mu_f'))
            elif variab == "def_rate":
                x_value = float(getParameterFromLatte(x+'/baseFiles/input.txt','defRate'))

            os.chdir(x)  # enter viscosity dir
            melt_labels = find_dirs()
            melt_labels.reverse()
            for m in melt_labels:
                if '_mR_0' in m:
                    melt_value = '0.00'+m.split('mR_0')[1]    # from full name of directory to 0.001, 0.002 etc
                else:
                    melt_value = '0.00'+m.split('mR0')[1]    # from full name of directory to 0.001, 0.002 etc
                # print(f'm {m}, melt_value {melt_value}')
                melt_number = melt_value.split("0")[-1]  # take the last value, ignore zeros
                file_number = str(round(t/(int(melt_number))/save_freq)*save_freq).zfill(5)  # normalise t by the melt rate.  .zfill(6) fills the string with 0 until it's 5 characters long
                print(f'file n normalised {file_number}')
                # melt_value = float(melt_value)
                data = read_and_filter_data_from_csv(m+'/my_experiment'+file_number+'.csv')
                if data.empty:
                    print("empty df") 
                    continue
                X = data[data['Broken Bonds'] > 0] # Only where there are bb 
                print(f'len X {len(X)}')
                #  if excluding margins:
                if no_margins:
                    X = X[X['x coord'] > 0.02] # exclude the left margin
                    X = X[X['x coord'] < 0.98] # # exclude the right margin 
                X = X[['x coord','y coord']] # Only x and y coordinates 
                if len(X) > 0:
                    df = dbscan_and_plot(X,0.0088,min_samples,df,variab,x_value,melt_value,no_margins)
            os.chdir('..')

            # now that it has read and calculated, save the data for next time
            df.to_csv(csv_name)
            print("saving")

    else:
        df = pd.read_csv(csv_name,index_col=0)
        print("reading")
        
    
    # cluster_variables = ['cluster_n','average_size','average_angle','average_elong']
    # for i,cluster_variable in enumerate(cluster_variables):
    #     plot_cluster_data(df,cluster_variable,cluster_variable_labels[i])
    cluster_variables = {'cluster_n': "Number of Clusters",
            'average_size':"Average Cluster Size",
            'average_angle':"Average Angle",
            'average_angle_from_90':"Average Deviation from 90 degrees",
            'average_elong':"Average Elongation"}

    if plot_figures:
        plt.show()  # show all of them at the end

    # prepare figure for plotting, only create these figures once
    fig = plt.figure(figsize=(8, 10))

    for cluster_variable in cluster_variables:
        """
        one big heatmap (scatterplot) for each variable 
        """
        plot_cluster_data(fig,df,cluster_variable,cluster_variables[cluster_variable],no_margins)
    # plot_cluster_data(fig,df,'cluster_n',"Number of Clusters")

    plt.close(fig)

    # make a grid with plots for each variable pair
    plot_pair_grid(df,t,no_margins)


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


parent_dir_path = 'rt0.5/'
# get the list of directories, viscosity dirs and melt rate dirs
os.chdir(parent_dir_path)

for t in times:
    cluster_analysis(t,no_margins)
print("Done!")