import numpy as np
from sklearn.cluster import DBSCAN
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import colorcet  as cc




def read_bb_data_from_csv(file_path):
    """ keep only x and y coordinates that have at least one broken bond """
    if os.path.isfile(file_path):
        all_data = pd.read_csv(file_path)
        bb_df = all_data[['x coord','y coord','Broken Bonds']]

        return bb_df
    else:
        return pd.DataFrame() 

def dbscan_and_plot(X,variab,x_value,melt_value,no_margins,plot_figures,t):
    min_samples = 10
    e = 0.0088
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
                palette = sns.color_palette(cc.glasbey_dark, n_colors=len(unique_labels))

                # scatter = plt.scatter(x_coords, y_coords, c=dbscan.labels_, marker='o', s=10, alpha=0.75,edgecolor='k',linewidth=edges)#, cmap=custom_colormap) #  edgecolor='k',
                scatter = sns.scatterplot(data=X_filtered,x='x coord',y='y coord',hue='cluster',palette=palette,marker='o',alpha=0.9,s=10,edgecolor='k',linewidth=0)
                
                plt.title('DBSCAN Clustering, eps ='+str(e)+', min_samples = '+str(min_samples) + ', min elements in a cluster = '+str(n_elements)+'\n'+os.getcwd()+" mr"+str(melt_value))
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
                # for index, row in centroids.iterrows():
                #     plt.text(row['x coord'], row['y coord'], f"{int(index)}", horizontalalignment='center', size='medium', color='black', weight='semibold')
                # plt.show()
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
                    if no_margins:
                        plt.savefig("../cluster/ellipses_x_v"+str(x_value)+"_mr"+str(melt_value)+"_"+str(t)+".png")  # show all of them at the end
                    else:
                        plt.savefig("../cluster/ellipses_v"+str(x_value)+"_mr"+str(melt_value)+"_real_t_"+str(t)+".png")  # show all of them at the end

                # Store the eigenvalues in the dictionary
                moments_of_inertia[cluster_label] = eigenvalues
                all_eigenvectors[cluster_label] = eigenvectors
                all_elong[cluster_label] = eigenvalues[0]/eigenvalues[1]
                all_angles[cluster_label] = angle_degrees
                all_sizes[cluster_label] = len(cluster_points)+1
            
            ellipses_df = pd.DataFrame({
                "Moments_of_Inertia": pd.Series(moments_of_inertia),
                "Eigenvectors": pd.Series(all_eigenvectors),
                "Elongation": pd.Series(all_elong),
                "Angle_Degrees": pd.Series(all_angles),
                "Size": pd.Series(all_sizes)
            })
            # print(ellipses_df)
            # if len(ellipses_df) > 2:
            #     anova_analysis(ellipses_df)

            #compute some global data
            weighted_sum = sum(angle * all_sizes[cluster] for cluster, angle in all_angles.items())
            weighted_average_angle = weighted_sum / sum(all_sizes.values())
            weighted_elong = sum(elong * all_sizes[cluster] for cluster, elong in all_elong.items())
            weighted_average_elong = weighted_elong / sum(all_sizes.values())

            cluster_data = {'cluster_n':len(all_sizes),'average_size':sum(all_sizes.values())/len(all_sizes),'average_angle':weighted_average_angle,
            'average_angle_from_90':abs(90-weighted_average_angle),'average_elong':weighted_average_elong,variab:x_value,'melt_rate':melt_value}

        else:
            """ clusters were found, but they are too small: populate the dataframe with NaN """
            cluster_data = fill_db_with_NaN(variab,x_value,melt_value)
            print("Clusters are too small")
            ellipses_df = {}

    else:
        """ no clusters were found: populate the dataframe with NaN """
        cluster_data = fill_db_with_NaN(variab,x_value,melt_value)

        ellipses_df = {}
        print("No Clusters")
        
    # add the new data to the existing dataframe
    cluster_df = pd.DataFrame(cluster_data,index=[0])
    # df = pd.concat([df,cluster_df], ignore_index=True) 
    
    return cluster_df,ellipses_df


def fill_db_with_NaN(variab,x_value,melt_value):
    cluster_data = {'cluster_n':float('NaN'),'average_size':float('NaN'),'average_angle':float('NaN'),
        'average_angle_from_90':float('NaN'),'average_elong':float('NaN'),variab:x_value,'melt_rate':melt_value}
    return cluster_data

def draw_rose_ellipse(ellipses_df):
    """
    get the orientations from the ellipse dataframe and plot them in a rose diagram
    proportional = if it should use weights when building the rose diagram
    """
    angles =  ellipses_df['Angle_Degrees']
    
    bins = np.arange(-5, 366, 10)
    lengths =  ellipses_df['Size']
        # print(f'lengths: {lengths}\nmin: {np.min(lengths)}, max: {np.max(lengths)}')
    if np.max(lengths)-np.min(lengths) != 0:
        # normalise lengths so that they go from 0 to 1
        lengths_norm = (lengths-np.min(lengths))/(np.max(lengths)-np.min(lengths))
    else:
        # if they are all the same length, or if there is only one, 
        # the normalised version of the array is 1 over their number (e.g. 1 in the case of 1 edge)
        lengths_norm = np.ones(len(lengths))/len(lengths) 
    # use lengths as weights for the histogram
    angles_in_bins, bins = np.histogram(angles, bins,weights=lengths_norm)


    # Sum the last value with the first value.
    angles_in_bins[0] += angles_in_bins[-1]

    # shouldn't be necessary, but in case there are angles > 180, sum them to their corresponding 
    # angle between 0 and 180. This way the result is symmetric: bottom half is the same 
    # as top half but mirrored
    single_half = np.sum(np.split(angles_in_bins[:-1], 2), 0)
    full_range = np.concatenate([single_half,single_half])  # repeat the sequence twice
    ax = plot_rose(full_range)

    return ax


def plot_rose(full_range: np.ndarray):
    # make plot
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='polar')

    ax.set_theta_zero_location('E') # zero starts at East
    # ax.set_theta_direction(-1)
    tens=np.arange(0, 360, 10)
    labels=[i if i%30==0 else '' for i in tens]
    ax.set_thetagrids(tens, labels=labels,weight='bold')
    ax.tick_params(axis="y", labelsize=11)
    ax.tick_params(axis="x", labelsize=13)
    ax.grid(linewidth=1.5)
    # ax.set_rgrids(np.arange(1, full_range.max() + 1, 2), angle=0, weight= 'black')
    # the height of each bar is the number of angles in that bin
    # ax.grid(False)
    ax.bar(np.deg2rad(np.arange(0, 360, 10)), full_range, 
        width=np.deg2rad(10), bottom=0.0, color=(0.5, 0.5, 0.5, 0.5), edgecolor='k')
        # width=np.deg2rad(10), bottom=0.0, color=(1, 0, 0), edgecolor='r')
    return ax