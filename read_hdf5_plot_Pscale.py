"""
read hdf5 files, take some of the dataframes in it and plot them

"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# run from myExperiments

def get_columns_to_plot(df):
    df_cols = []
    for col in df.columns:
        df_cols.append(col)
    return df_cols[::200]  # extract every 200 tsteps

def apply_scale(coord):
    """ applies a scaling and translation to the x coordinate """
    if scale == '050':
        scale_shift = 0.125
    elif scale == '200':
        scale_shift = 0.5
    
    scale_factor = 200/float(scale)  # e.g. 200/50 = 4
    return (coord-scale_shift)#/scale_factor # do the scaling  = (0.125-0.125)/4 


dfm_old = pd.DataFrame()
scales = ['050','200']
for scale in scales:
    if scale == '200':
        h5 = "/nobackup/scgf/myExperiments/gaussTime/my_h5_copy.h5"
    elif scale == '050':
        h5 = "/nobackup/scgf/myExperiments/gaussScale/scale050/my_h5_copy2.h5"
    
    store = pd.HDFStore(h5)
    my_keys = store.keys()
    df = store[my_keys[-1]]
    cols_to_plot = get_columns_to_plot(df)  # get a subset of columns
    dfm = df.melt('x_coordinate', value_vars=cols_to_plot, var_name='time', value_name='Fluid Pressure') # reshape df so that it has 1 column for x coord and 1 for pressure
    # https://pandas.pydata.org/docs/reference/api/pandas.melt.html
    dfm['scale'] = scale  # add a column to store info on scale
    #print(dfm)
    #assert apply_scale(0.125) == 0.0, "Should be 0.0"
    dfm['new_x_coord'] = dfm['x_coordinate'].apply(apply_scale)
    dfm = pd.concat([dfm_old,dfm],ignore_index=True)
    dfm_old = dfm

g = sns.lineplot(x="new_x_coord", y='Fluid Pressure', hue='time',style='scale', data=dfm)#, kind='point',s = 1)

# for my_key in store200.keys():
#     df = store200[my_key]
    #ax1 = sns.lineplot(data=df_sigma_1_0_gaussTime02_t1e3_scale200,x='x_coordinate',y='t_20000')
# /sigma_1_0/gaussTime02/t1e3

# df = store200[my_keys[-1]]
# df1 = store050[my_keys[-1]]

# # convert to long (tidy) form
# dfm = df.melt('x_coordinate', var_name='time', value_name='vals')
# dfm1 = df.melt('x_coordinate', var_name='time', value_name='vals')
# g = sns.lineplot(x="x_coordinate", y="vals", hue='time', data=dfm)#, kind='point',s = 1)
# # sns.lineplot(data=fmri, x="timepoint", y="signal", hue="region", style="event")
# sns.lineplot(x="x_coordinate", y="vals", hue='time', data=dfm1,ax=g)#, kind='point',s = 1)

# df_sigma_1_0_gaussTime02_t1e3_scale050 = store200["/sigma_1_0/gaussTime02/t1e4"]
# df_sigma_1_0_gaussTime02_t1e3_scale200 = store050['/sigma_1_0/gaussTime02/t1e4']
# for col_name in df_sigma_1_0_gaussTime02_t1e3_scale200.columns:
#     print(col_name)
# ax1 = sns.lineplot(data=df_sigma_1_0_gaussTime02_t1e3_scale200,x='x_coordinate',y='t_20000')
# sns.lineplot(data=df_sigma_1_0_gaussTime02_t1e3_scale200,x='x_coordinate',y='t_20000',ax=ax1)
#g.set_xlim(0.075,0.2)
#plt.xlim(0.00125,0.98875)
#plt.xlim(150, 250)
plt.xlim(-0.10, +0.10)
#plt.xticks(np.arange(150,10,250))
plt.show()