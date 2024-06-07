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
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import pearsonr

sys.path.append('/home/home01/scgf/myscripts/post_processing')

from useful_functions import getSaveFreq,getParameterFromLatte
from dbscan_functions import read_bb_data_from_csv,dbscan_and_plot,fill_db_with_NaN
from viz_functions import find_dirs,find_variab

save_freq = int(getSaveFreq())
no_margins = 1

# times = list(range(1, 20, 1)) + list(range(20, 141, 5)) + list(range(150, 501, 20)) + list(range(500, 801, 20)) + list(range(850, 1500, 40))
# times = [88,100,300]
times = [79]
times = [i*save_freq for i in times]   # convert to timestep


plot_figures = True  # if I want to visualise clusters, ellipses etc





def plot_cluster_data(fig,df,cluster_variable,cluster_variable_label,no_margins):
    # set which figure to use
    plt.figure(fig)
    plt.clf()

    # prepare to handle NaNs
    mask_nan = df[cluster_variable].isna()

    # prepare to highlight big values
    # smallest_value = df[cluster_variable].min()
    # big_value = (df[cluster_variable].max() - smallest_value) * 3/4 + smallest_value
    # print(f'{cluster_variable_label}: big_value {big_value}, max {df[cluster_variable].max()}, min {df[cluster_variable].min()}')
    # df_big_values_mask = df[cluster_variable] > big_value


    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 4], height_ratios=[2, 7])
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
    square_size = 2600
    ax0 = plt.subplot(gs[1, 0])
    # sns.heatmap(heatmap_data, ax=ax0, cmap="viridis", cbar_kws={'label': 'Average Size'})
    sns.scatterplot(data=df, x='viscosity',y='melt_rate',hue=cluster_variable,
                    marker='s',s=square_size,edgecolor='none', ax=ax0, palette="flare")
    ax0.set_xlabel("Viscosity")
    ax0.set_ylabel("Melt Rate")
    ax0.set_xscale('log')

    # Scatter plot for NaN values 
    sns.scatterplot(data=df[mask_nan], x='viscosity', y='melt_rate', color='gray',
                    marker='s', s=square_size,edgecolor='none', ax=ax0)

    # # highlight big values
    # sns.scatterplot(data=df[df_big_values_mask], x='viscosity', y='melt_rate',facecolors='none', hatch='///',
    #                 marker='s',edgecolor='black', s=square_size*0.6, ax=ax0)

    df = df.sort_values(by=['melt_rate','viscosity'])


    def create_palette(base_colors, n_colors):   # Function to create a color palette
        colormap = LinearSegmentedColormap.from_list("custom_palette", base_colors, N=n_colors)
        # Generate colors from the colormap
        return [to_hex(colormap(i / (n_colors - 1))) for i in range(n_colors)]

    palette_visc = create_palette(["#aae88e","#397367", "#140021"], len(df['viscosity_label'].unique()))
    palette_visc_dict = dict(zip(df['viscosity_label'].unique(), palette_visc))

    palette_mr = create_palette(["#c2823a", "#33231E"], len(df['melt_rate'].unique()))
    palette_mr_dict = dict(zip(df['viscosity_label'].unique(), palette_mr))

    # Top plot (Viscosity vs Average Size)
    ax1 = plt.subplot(gs[0, 0], sharex=ax0)   # sharex  so they are aligned
    sns.lineplot(data=df, x="viscosity", y=cluster_variable,hue="melt_rate", ax=ax1, palette=palette_mr)
    sns.scatterplot(data=df, x="viscosity", y=cluster_variable,hue="melt_rate", ax=ax1, palette=palette_mr,legend='')
    ax1.set_xlabel("")
    ax1.set_ylabel(cluster_variable_label)
    # ax1.legend(title="Melt Rate",ncol=1, fontsize=7, title_fontsize=9)
    ax1.xaxis.tick_top()
    # Invert the order in the legend (small at the bottom)
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles[::-1], labels[::-1],title="Melt Rate",fontsize=7)


    # Right plot (Melt Rate vs Average Size)

    ax2 = plt.subplot(gs[1, 1])#, sharey=ax0)    # sharey  so they are aligned
    sns.lineplot(data=df, x="melt_rate", y=cluster_variable,hue="viscosity_label", ax=ax2, palette=palette_visc_dict)
    sns.scatterplot(data=df, x="melt_rate", y=cluster_variable,hue="viscosity_label", ax=ax2, palette=palette_visc_dict,legend='')
    ax2.set_xlabel("Melt Rate")
    # ax2.set_yticks("")  # no ticks - it removes them from the heatmap too
    ax2.set_ylabel(cluster_variable_label)
    # ax2.legend(title="$\mu_f$", fontsize=7, title_fontsize=9)
    ax2.yaxis.tick_right()
    # Invert the order in the legend (small at the bottom)
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles[::-1], labels[::-1],title="$\mu_f$",fontsize=7)

    # Show or save the plot
    # plt.show()
    
    if no_margins:  # excluding margins, use a different name
        plt.savefig("cluster/cluster_x_"+cluster_variable+"_"+str(t)+".png")
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

    # hue is viscosity
    grid_mu = sns.PairGrid(df_pair.drop(columns=['melt_rate']), hue='viscosity', palette="crest") # (df.dropna())
    grid_mu = grid_mu.map_offdiag(sns.scatterplot)
    grid_mu = grid_mu.map_diag(sns.histplot)
    grid_mu.add_legend(title="Viscosity")
    
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
    if no_margins:
        plt.savefig("cluster/cluster_x_grid_mr"+str(t)+".png")
    else:
        plt.savefig("cluster/cluster_grid_mr"+str(t)+".png")
    
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
                print(f'file n normalised {file_number}')
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
                    cluster_df = dbscan_and_plot(X,variab,x_value,melt_value,no_margins,plot_figures,t)
                else:
                    # no fractures at all: we still need to have an entry in the dataframe, but it will be NaN
                    cluster_data = fill_db_with_NaN(variab,x_value,melt_value)
                    cluster_df = pd.DataFrame(cluster_data,index=[0])
                df = pd.concat([df,cluster_df], ignore_index=True) 
            os.chdir('..')

            # now that it has read and calculated, save the data for next time
        df.to_csv(csv_name)
        print("saving")

    else:
        df = pd.read_csv(csv_name,index_col=0)

            
        print("reading")
        
    if variab == "viscosity":
        df["viscosity"] = df["viscosity"]*1000

    print(df)

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
        plot_cluster_data(fig,df,cluster_variable,cluster_variables[cluster_variable],no_margins)
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