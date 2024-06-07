"""
calculate u*phi as a measure of flux leaving the domain.
ratio between y and x direction to get the flux anisotropy

"""

import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import sys
import pandas as pd
import seaborn as sns
import matplotlib.colors as mcolors
sys.path.append('/home/home01/scgf/myscripts/post_processing')
from useful_functions import extract_two_profiles,getResolution,getParameterFromLatte,getSaveFreq
from viz_functions import find_dirs,find_variab



res = getResolution()    # resolution in the x direction
res_y = int(res*1.15)  # resolution in the y direction

# define the indexes for the two profiles, the first one is the start for the horizontal, the second for the vertical profile
# REMEMBER to remove 1 from the horizontal index
point_indexes = [(res*res_y-(5*res))-1,5]   # top (horizontal) and left (vertical). X original: res*res_y/2+res, central: res*res_y/2 (?)
# point_indexes = [(res*res_y*0.75+res),5]   # 3/4 of the domain
# print(f'index for the horizontal profile: {point_indexes[0]}')



fig, ax1 = plt.subplots(nrows=1, ncols=1)

x_variable = find_dirs()  # viscosity dirs
variab = find_variab()
save_freq = int(getSaveFreq())
t = 100*save_freq


vars_to_plot = ["x velocity","y velocity", "Porosity"]
df = pd.DataFrame()
csv_name = 'flow_anisotropy_t'+str(t)+'.csv'

if not os.path.isfile(csv_name):

    for x in x_variable:
        print(x)
        # if x == 'visc_2_1e2':
        if variab == "viscosity":
            x_value = float(getParameterFromLatte(x+'/baseFiles/input.txt','mu_f'))*1000
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
            filename = m+"/"+'my_experiment'+file_number+'.csv'
            print(filename)
            print(os.getcwd())

            # prepare to store vertical and horizontal data
            all_data_v = {}
            all_data_h = {}
            if not os.path.isfile(filename):
                integral_data = {'integral_ver':float('NaN'),'integral_hor':float('NaN'),'int_ratio':float('NaN'),
                    variab:x_value,'melt_rate':melt_value}
            else:
                for v in vars_to_plot:
                    (x_v, y_v), (x_h, y_h) = extract_two_profiles(filename, v,point_indexes)
                    all_data_v[v] = (x_v, y_v)
                    all_data_h[v] = (x_h, y_h)
                    

                x_coord_vel_hor, y_vel_hor = all_data_h[vars_to_plot[1]] 
                x, poro_values_hor = all_data_h[vars_to_plot[2]]


                integral_hor = np.trapz((y_vel_hor), x=x_coord_vel_hor)
                print(f'integral_hor {integral_hor}')

                y_coord_vel_ver, x_vel_ver = all_data_v[vars_to_plot[0]] 
                x, poro_values_ver = all_data_v[vars_to_plot[2]]
                integral_ver =  np.trapz((x_vel_ver*poro_values_ver), x = y_coord_vel_ver)

                int_ratio = integral_hor/integral_ver

                print(f'integral_ver {integral_ver}')
                print(f'integral ratio {int_ratio}')
                integral_data = {'integral_ver':integral_ver,'integral_hor':integral_hor,'int_ratio':int_ratio,
                    variab:x_value,'melt_rate':melt_value}

            # add the new data to the existing dataframe
            new_data_df = pd.DataFrame(integral_data,index=[0])
            df = pd.concat([df,new_data_df], ignore_index=True) 

        os.chdir('..')
else:
    print("reading")
    df = pd.read_csv(csv_name,index_col=0)

print(df)

df.to_csv(csv_name)

min_int = df['int_ratio'].min()
max_int = df['int_ratio'].max()
print(f'min int ratio: {min_int}, max int ratio: {max_int}')

norm = mcolors.Normalize(vmin=-20, vmax=20)

square_size = 2600
sns.scatterplot(data=df, x='viscosity',y='melt_rate',hue='int_ratio',
                marker='s',s=square_size,edgecolor='none', ax=ax1, palette="flare",norm=norm)#,vmin=-20,vmax=20)
ax1.set_xlabel("Viscosity")
ax1.set_ylabel("Melt Rate")
ax1.set_xscale('log')

# mask_nan = df['int_ratio'].isna()

# Scatter plot for NaN values 
# sns.scatterplot(data=df[mask_nan], x='viscosity', y='melt_rate', color='gray',
#                 marker='s', s=square_size,edgecolor='none', ax=ax1)

                
plt.show()

