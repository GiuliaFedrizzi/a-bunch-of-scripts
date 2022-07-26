import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# run from gaussTime/
h5_200 = "my_h5.h5"
h5_050 = "../gaussScale/scale050/my_h5.h5"


store200 = pd.HDFStore(h5_200)
store050 = pd.HDFStore(h5_050)

# /sigma_1_0/gaussTime02/t1e3
df_sigma_1_0_gaussTime02_t1e3_scale050 = store200["/sigma_1_0/gaussTime02/t1e3"]
df_sigma_1_0_gaussTime02_t1e3_scale200 = store050['/sigma_1_0/gaussTime02/t1e3']
for col_name in df_sigma_1_0_gaussTime02_t1e3_scale200.columns:
    print(col_name)
sns.lineplot(data=df_sigma_1_0_gaussTime02_t1e3_scale200,x='x_coordinate',y='t_20000')
plt.show()