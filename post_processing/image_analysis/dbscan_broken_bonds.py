"""
Read my_experiment*.csv corresponding to the (normalised) time
perform DBSCAN analysis on fracture locations
analysis on clusters:
- elongation
- number of clusters
- cluster size
- orientation

To be run from outside of rt0.5/

Giulia, May 2024
"""

# import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import csv
from scipy.stats import gaussian_kde
from scipy.spatial.distance import cdist
from matplotlib.colors import ListedColormap,LinearSegmentedColormap, to_hex
from matplotlib.patches import Ellipse
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
import pandas as pd
import sys
import os
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
# import statsmodels.api as sm
# from statsmodels.formula.api import ols
from scipy.stats import pearsonr

sys.path.append('/Users/giuliafedrizzi/Library/CloudStorage/OneDrive-UniversityofLeeds/PhD/scripts/a-bunch-of-scripts2/a-bunch-of-scripts/post_processing')

from useful_functions import getSaveFreq,getParameterFromLatte
from dbscan_functions import read_bb_data_from_csv,dbscan_and_plot,fill_db_with_NaN,draw_rose_ellipse
from viz_functions import find_dirs,find_variab

save_freq = int(getSaveFreq())
no_margins = 1

# times = list(range(1, 20, 1)) + list(range(20, 141, 5)) + list(range(150, 501, 20)) + list(range(500, 801, 20)) + list(range(850, 1500, 40))
# times = [88,100,300]
times = [100]
times = [i*save_freq for i in times]   # convert to timestep


plot_figures = True  # if I want to visualise clusters, ellipses etc


plt.rcParams["font.weight"] = "bold"
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
        # plt.rcParams['ytick.labelsize'] = 15
        # plt.rcParams['ytick.minor.size'] = 4
        # plt.rcParams['ytick.minor.width'] = 2
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['xtick.major.width'] = 3
plt.rcParams['ytick.major.size'] = 5
plt.rcParams['ytick.major.width'] = 3
plt.rcParams['xtick.minor.size'] = 4
plt.rcParams['xtick.minor.width'] = 2

def plot_cluster_data(fig,df,cluster_variable,cluster_variable_label,no_margins,variab):
    # set which figure to use
    plt.figure(fig)
    plt.clf()

    # prepare to handle NaNs
    mask_nan = df[cluster_variable].isna()

    gs = gridspec.GridSpec(2, 2, width_ratios=[6,5], height_ratios=[3, 7])
    fig.suptitle(cluster_variable_label+", t = " + str(t))
    print(f'cluster_variable_label {cluster_variable_label}')
    ax0 = plt.subplot(gs[1, 0])
     
    if variab == "viscosity":
        unique_viscosity_values = set(df['viscosity'])
        ax0.set_xlabel("Viscosity")


        #prepare labels
        viscosity_labels = {}
        for v in unique_viscosity_values:
            viscosity_labels[v] = "$10^{"+str(np.log10(v))+"}$"
            df['viscosity_label'] = df['viscosity'].map(viscosity_labels)
            variable_labels = 'viscosity_label'
        ax0.set_xscale('log')
    else:
        ax0.set_xlabel("Deformation Rate")
        variable_labels = 'def_rate'
    df['melt_rate'] = df['melt_rate'].astype(str)

    # pseudo heatmap
    square_size = 3200
    marker_scale_legend = 0.2  # how much smaller the item in the legend is
    # sns.heatmap(heatmap_data, ax=ax0, cmap="viridis", cbar_kws={'label': 'Average Size'})
    sns.scatterplot(data=df, x=variab,y='melt_rate',hue=cluster_variable,
                    marker='s',s=square_size,edgecolor='none', ax=ax0, palette="flare")
    
    legend_marker_size = square_size*marker_scale_legend*0.1
    handles, labels = ax0.get_legend_handles_labels()

    print(f'handles {handles}, labels {labels}')

    # Add custom handle for "No clusters"
    
    no_clusters_handle = Line2D([0], [0], marker='s', color='w', label='No clusters',
                                markerfacecolor='gray', markersize=legend_marker_size, linestyle='None')
    handles.append(no_clusters_handle)
    labels.append('No clusters')

    print(f'handles {handles}, labels {labels}')

    legend_title = cluster_variable_label
    if cluster_variable_label == "Number of Clusters":
        legend_title = "Number\nof Clusters"
    elif cluster_variable_label == "Average Cluster Size":
       legend_title = "Average\nCluster Size"
    elif cluster_variable_label == "Average Deviation from 90 degrees":
        legend_title = "Average\nDeviation from 90°"
    ax0.legend(handles=handles, labels=labels, markerscale=marker_scale_legend, title=legend_title, fontsize=11, title_fontsize=12, loc='lower left', frameon=True)


    ax0.set_ylabel("Melt Rate")

    # Scatter plot for NaN values 
    sns.scatterplot(data=df[mask_nan], x=variab, y='melt_rate', color='gray',
                    marker='s', s=square_size,edgecolor='none', ax=ax0, legend=False)

    ax0.set_xlim([6e3, 5e6])  # set xlim to include an extra half square

    plot_df = df[df[cluster_variable].notna()]
    df = df.sort_values(by=['melt_rate',variab])
    plot_df = plot_df.sort_values(by=['melt_rate',variab])


    def create_palette(base_colors, n_colors):   # Function to create a color palette
        colormap = LinearSegmentedColormap.from_list("custom_palette", base_colors, N=n_colors)
        # Generate colors from the colormap
        return [to_hex(colormap(i / (n_colors - 1))) for i in range(n_colors)]

    palette_visc = create_palette(["#aae88e","#397367", "#140021"], len(plot_df[variable_labels].unique()))
    palette_visc_dict = dict(zip(plot_df[variable_labels].unique(), palette_visc))

    palette_mr = create_palette(["#c2823a", "#33231E"], len(plot_df['melt_rate'].unique()))
    palette_mr_dict = dict(zip(plot_df[variable_labels].unique(), palette_mr))

    # Top plot (Viscosity vs Average Size)
    ax1 = plt.subplot(gs[0, 0], sharex=ax0)   # sharex  so they are aligned
    sns.lineplot(data=plot_df, x=variab, y=cluster_variable,hue="melt_rate", ax=ax1, palette=palette_mr)
    sns.scatterplot(data=plot_df, x=variab, y=cluster_variable,hue="melt_rate", ax=ax1, palette=palette_mr,legend='')
    ax1.set_xlabel("")
    ax1.set_ylabel(cluster_variable_label)
    if cluster_variable_label == "Number of Clusters":
        ax1.set_ylabel("Number\nof Clusters")
    elif cluster_variable_label == "Average Cluster Size":
        ax1.set_ylabel("Average\nCluster Size")
    elif cluster_variable_label == "Average Deviation from 90 degrees":
        ax1.set_ylabel("Average\nDeviation from 90°")

    # ax1.legend(title="Melt Rate",ncol=1, fontsize=7, title_fontsize=9)
    ax1.xaxis.tick_top()
    # Invert the order in the legend (small at the bottom)
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles[::-1], labels[::-1],title="Melt Rate",fontsize=11, title_fontsize=12)
    ax1.grid(linestyle='--',alpha=0.6)

    # Right plot (Melt Rate vs Average Size)

    ax2 = plt.subplot(gs[1, 1])#, sharey=ax0)    # sharey  so they are aligned
    sns.lineplot(data=plot_df, x="melt_rate", y=cluster_variable,hue=variable_labels, ax=ax2, palette=palette_visc_dict)
    sns.scatterplot(data=plot_df, x="melt_rate", y=cluster_variable,hue=variable_labels, ax=ax2, palette=palette_visc_dict,legend='')
    ax2.set_xlabel("Melt Rate")
    # ax2.set_yticks("")  # no ticks - it removes them from the heatmap too
    ax2.set_xlim([-1.65, 7.35])  # set xlim to include the original values, even if they are not in plot_df
    plt.draw()
    ticks = ax2.get_xticks()  # ticks are: [0, 1, 2, 3, 4, 5, 6, 7, 8]  -> they correspond to [0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009]  !! note that the first tick (0) is 0.002 ! 
    ticks = np.append(-1,ticks)
    new_labels = ["" if i % 2 == 0 else (i+2)/1000 for i in ticks]  #  keep only every second label: [0.001, '', 0.003, '', 0.005, '', 0.007, '', 0.009]  i+2 because it starts counting from 0 and the first tick is 0.002
    ax2.set_xticks(ticks)
    ax2.set_xticklabels(new_labels)

    ax2.set_ylabel(cluster_variable_label)
    if cluster_variable_label == "Average Deviation from 90 degrees":
        ax2.set_ylabel("Average Deviation from 90°")
    ax2.yaxis.set_label_position("right")
    # ax2.legend(title="$\mu_f$", fontsize=7, title_fontsize=9)
    ax2.yaxis.tick_right()
    # Invert the order in the legend (small at the bottom)
    handles, labels = ax2.get_legend_handles_labels()
    if variab == "viscosity":
        legend_title = "$\mu_f$"
    else:
        legend_title = "Def rate"

    plt.subplots_adjust(wspace=0.03,hspace=0.03)  # adjust space between subplots

    ax2.legend(handles[::-1], labels[::-1],title=legend_title,fontsize=11, title_fontsize=12)
    ax2.grid(linestyle='--',alpha=0.6)

    # Show or save the plot
    # plt.show()
    
    if no_margins:  # excluding margins, use a different name
        plt.savefig("cluster/cluster_x1_"+cluster_variable+"_"+str(t)+".png")
    else:
        plt.savefig("cluster/cluster_"+cluster_variable+"_"+str(t)+".png")



def plot_pair_grid(df,t,no_margins):
    """
    make a grid with plots for each variable pair.
    Use melt rate and viscosity as hue (one figure for each hue)
    """

    fig_pair = plt.figure(figsize=(7, 7))
    # plt.title("t = "+str(t))

    df_pair = df.drop(columns=['average_angle','viscosity_label'])   # keep only the columns that are useful for the grid plot
    df_pair = df_pair.dropna()

    # set which figure to use
    plt.figure(fig_pair)

    if 'viscosity' in df.columns:
        visc_or_def = 'viscosity'
    elif 'def_rate' in df.columns:
        visc_or_def = 'def_rate'
    # hue is viscosity or melt rate
    grid_mu = sns.PairGrid(df_pair.drop(columns=['melt_rate']), hue=visc_or_def, palette="crest") # (df.dropna())
    grid_mu = grid_mu.map_offdiag(sns.scatterplot)
    grid_mu = grid_mu.map_diag(sns.histplot)
    if visc_or_def == 'viscosity':
        grid_mu.add_legend(title="Viscosity")
    elif visc_or_def == 'def_rate':
        grid_mu.add_legend(title="Deformation Rate")
    
    #  if excluding margins:
    if no_margins:
        plt.savefig("cluster/cluster_x_grid_visc"+str(t)+".png")
    else:
        plt.savefig("cluster/cluster_grid_visc"+str(t)+".png")
    plt.clf()  #  clear the figure, ready to be reused

    # plt.title("t = "+str(t))

    # hue is melt rate
    grid_mr = sns.PairGrid(df_pair.drop(columns=['viscosity']), hue='melt_rate', palette="crest_r") # (df.dropna())
    grid_mr = grid_mr.map_offdiag(sns.scatterplot)
    grid_mr = grid_mr.map_diag(sns.histplot)
    grid_mr.add_legend(title="Melt Rate")
    
    #   if excluding margins:
    # if no_margins:
    #     plt.savefig("cluster/cluster_x_grid_mr"+str(t)+".png")
    # else:
    #     plt.savefig("cluster/cluster_grid_mr"+str(t)+".png")
    
    plt.close(fig_pair)  # we're done with the figures

    # df_pair = 
    print(f'df pair before correl')
    print(df_pair)

    # def anova_formula(df_pair,formula):
    #     # calculate anova between viscosity or melt rate and the other variables
    #     model = ols(formula, data=df_pair).fit()
    #     anova_results = sm.stats.anova_lm(model, typ=2)

    #     # Print the results
    #     print(anova_results)

    # # create all combinations:
    # pairs = []
    # for var in ['melt_rate','viscosity']:
    #     for column_2 in df_pair:
    #         if var != column_2:
    #             if [var,column_2] not in pairs:
    #                 formula = f'{var} ~ {column_2}'
    #                 print(formula)
    #                 anova_formula(df_pair,formula)


    def plot_correl(df_pair):
        # correlation and p-values
        corr = df_pair.corr()
        p_values = pd.DataFrame(data=[[pearsonr(df_pair[col], df_pair[col2])[1] for col2 in df_pair.columns] for col in df_pair.columns], columns=df_pair.columns, index=df_pair.columns)
        print(p_values)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)

        # add the p-values to the heatmap
        for i in range(len(corr.columns)):
            for j in range(len(corr.columns)):
                plt.text(i+0.5, j+0.1, 'p = {:.3f}'.format(p_values.iloc[i, j]),
                        horizontalalignment='center',
                        verticalalignment='center', 
                        color="black" if corr.iloc[i, j] > 0 else "white")

        plt.xticks(np.arange(0.5, len(df_pair.columns), 1), df_pair.columns, rotation=45, ha="right")
        plt.yticks(np.arange(0.5, len(df_pair.columns), 1), df_pair.columns)
        plt.title("Correlation Matrix with P-values")
        plt.show()

    df_pair = df_pair.drop(columns=['melt_rate','viscosity'])#,'average_angle'])
    df_pair = df_pair.dropna()
    print(df_pair)
    correlation_matrix = df_pair.corr(method='pearson')  # alternatives: 'spearman' or 'kendall'
    print("correlation_matrix")
    print(correlation_matrix)
    plot_correl(df_pair)

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
    variab = find_variab()

    # check if there is already a file with the data. If there isn't, read and calculate
    if not os.path.isfile(csv_name):
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
                # print(f'file n normalised {file_number}')
                # melt_value = float(melt_value)
                data = read_bb_data_from_csv(m+'/my_experiment'+file_number+'.csv')
                if data.empty:
                    print("empty df") 
                    continue
                X = data[data['Broken Bonds'] > 0] # Only where there are bb 
                # print(f'len X {len(X)}')
                #  if excluding margins:
                if no_margins:
                    X = X[X['x coord'] > 0.02] # exclude the left margin
                    X = X[X['x coord'] < 0.98] # # exclude the right margin 
                X = X[['x coord','y coord']] # Only x and y coordinates 
                if len(X) > 0:
                    cluster_df,ellipses_df,e = dbscan_and_plot(X,variab,x_value,melt_value,no_margins,plot_figures,t)
                    if len(ellipses_df) > 0:
                        # draw a rose diagram based on the ellipse orientation. This is weighted by the ellipse's size and elongation
                        ax_ell = draw_rose_ellipse(ellipses_df)
                        #  t is the reference time, not the true (normalised) time. This is different in dbscan_single, where it is the true time
                        plt.savefig("../cluster/rose_"+str(x_value)+"_"+str(melt_value)+"_ref_t_"+str(t)+"_e_"+str(e)+".png",transparent=True)

                else:
                    # no fractures at all: we still need to have an entry in the dataframe, but it will be NaN
                    cluster_data = fill_db_with_NaN(variab,x_value,melt_value)
                    cluster_df = pd.DataFrame(cluster_data,index=[0])
                # print(cluster_df)
                df = pd.concat([df,cluster_df], ignore_index=True) 
            os.chdir('..')

            # now that it has read and calculated, save the data for next time
        # df.to_csv(csv_name)
        # print("saving")

    else:
        df = pd.read_csv(csv_name,index_col=0)

            
        print("reading")
        
    if variab == "viscosity":
        df["viscosity"] = df["viscosity"]*1000


    cluster_variables = {'cluster_n': "Number of Clusters",
            'average_size':"Average Cluster Size",
            'average_angle':"Average Angle",
            'average_angle_from_90':"Average Deviation from 90 degrees",
            'average_elong':"Average Elongation"}

    # if plot_figures:
    #     plt.show()  # show all of them at the end

    
    # prepare figure for plotting, only create these figures once
    fig = plt.figure(figsize=(11, 10))

    for cluster_variable in cluster_variables:
    # one big heatmap (scatterplot) for each variable 
        print(f'cluster_variable {cluster_variable}')
        plot_cluster_data(fig,df,cluster_variable,cluster_variables[cluster_variable],no_margins,variab)
        plt.clf()
    # plot_cluster_data(fig,df,'cluster_n',"Number of Clusters")

    plt.close(fig)

    # make a grid with plots for each variable pair
    # plot_pair_grid(df,t,no_margins)
    
    

    if False:
        min_samples = 10
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

if not os.path.exists('cluster'):
    os.makedirs('cluster')

for t in times:
    cluster_analysis(t,no_margins)
print("Done!")