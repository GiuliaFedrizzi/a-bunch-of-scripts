"""
read hdf5 files, take some of the dataframes in it and plot them

"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# run from gaussTime/
h5_200 = "my_h5_copy.h5"
h5_050 = "../gaussScale/scale050/my_h5_copy.h5"


store200 = pd.HDFStore(h5_200)
store050 = pd.HDFStore(h5_050)

# for my_key in store200.keys():
#     df = store200[my_key]
    #ax1 = sns.lineplot(data=df_sigma_1_0_gaussTime02_t1e3_scale200,x='x_coordinate',y='t_20000')
# /sigma_1_0/gaussTime02/t1e3
my_keys = store200.keys()
df = store200[my_keys[-1]]
df1 = store050[my_keys[-1]]

# convert to long (tidy) form
dfm = df.melt('x_coordinate', var_name='time', value_name='vals')
dfm1 = df.melt('x_coordinate', var_name='time', value_name='vals')
g = sns.lineplot(x="x_coordinate", y="vals", hue='time', data=dfm)#, kind='point',s = 1)
# sns.lineplot(data=fmri, x="timepoint", y="signal", hue="region", style="event")
sns.lineplot(x="x_coordinate", y="vals", hue='time', data=dfm1,ax=g)#, kind='point',s = 1)

# df_sigma_1_0_gaussTime02_t1e3_scale050 = store200["/sigma_1_0/gaussTime02/t1e4"]
# df_sigma_1_0_gaussTime02_t1e3_scale200 = store050['/sigma_1_0/gaussTime02/t1e4']
# for col_name in df_sigma_1_0_gaussTime02_t1e3_scale200.columns:
#     print(col_name)
# ax1 = sns.lineplot(data=df_sigma_1_0_gaussTime02_t1e3_scale200,x='x_coordinate',y='t_20000')
# sns.lineplot(data=df_sigma_1_0_gaussTime02_t1e3_scale200,x='x_coordinate',y='t_20000',ax=ax1)
#g.set_xlim(0.075,0.2)
#plt.xlim(0.00125,0.98875)
#plt.xlim(150, 250)
plt.xlim(0.4, 0.6)
#plt.xticks(np.arange(150,10,250))
plt.show()