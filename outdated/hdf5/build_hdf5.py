"""
go through directories and make a single hdf5 file
Giulia July 2022
"""
# compares results from different sims in different directories (gaussRes) - pressure input with gaussian distribution 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import glob
import time
from multiprocessing import Pool # to run in parallel
import functools as ft
# ~~~ start time ~~~
sta = time.time()

# some variables for the subplots:
fig, axs = plt.subplots(nrows=1, ncols=1)

# initialise stuff
max_P = 0.0
min_P = 0.0

if os.path.exists('my_h5_limitFiles0_sigma3_par.h5'):
    os.remove('my_h5_limitFiles0_sigma3_par.h5')


# wrap csv importer in a function that can be mapped
def read_extract_P(file):
    """converts a filename to a pandas dataframe, gets pressure"""
        # go through files, one by one
    myExp = pd.read_csv(file, header=0)
    # find the location in the y coordinate corresponding to the max pressure 
    ymax = myExp.loc[myExp['Pressure'].idxmax(), 'y coord']
    # find the x coordinates that corresponds to the max pressure
    #x_array = np.array(myExp.loc[myExp.apply(lambda x: math.isclose(x['y coord'],ymax,rel_tol=1e-3),axis=1),'x coord'])
    # extract the pressure arrays
    pressure_array_x = np.array(myExp.loc[myExp.apply(lambda x: math.isclose(x['y coord'],ymax,rel_tol=1e-3),axis=1),'Pressure'])
    #pressure_array_y = np.array(myExp.loc[myExp.apply(lambda x: math.isclose(x['x coord'],xmax,rel_tol=1e-3),axis=1),'Pressure'])
    col_name = file.replace(".csv","").replace("my_experiment","t_")
    #df["x_coordinate"] = x_array
    df[col_name] = pressure_array_x  # store x coordinates
    return df

def read_extract_x_coord(file):
    """converts a filename to a pandas dataframe, gets x_array (x coordinate)"""
        # go through files, one by one
    myExp = pd.read_csv(file, header=0)
    # find the location in the y coordinate corresponding to the max pressure 
    ymax = myExp.loc[myExp['Pressure'].idxmax(), 'y coord']
    # find the x coordinates that corresponds to the max pressure
    x_array = np.array(myExp.loc[myExp.apply(lambda x: math.isclose(x['y coord'],ymax,rel_tol=1e-3),axis=1),'x coord'])
    # extract the pressure arrays
    #pressure_array_x = np.array(myExp.loc[myExp.apply(lambda x: math.isclose(x['y coord'],ymax,rel_tol=1e-3),axis=1),'Pressure'])
    #pressure_array_y = np.array(myExp.loc[myExp.apply(lambda x: math.isclose(x['x coord'],xmax,rel_tol=1e-3),axis=1),'Pressure'])
    x_series = pd.Series(x_array, name="x coordinate")
    return x_series



for sigma_dir in sorted(glob.glob("sigma_3*")): # sigma_1_0
    print(sigma_dir)
    os.chdir(sigma_dir)
    time0_dirs = sorted(glob.glob("sigma_3*"))

    for time0_dir in time0_dirs:
        """ sigma_1_0_gaussTime02, ...time04, ...time05  for loop """
        os.chdir(time0_dir)
        print("*** Entering " + str(os.getcwd()))
        #dirList = []
        # for my_dir in sorted(glob.glob("tstep*")):
        #     """ get the list of directories"""
        #     dirList.append(my_dir)
        # print(dirList)
        
        for tstep_dir in sorted(glob.glob("tstep*")):   # e.g. tstep02_3_5e3
            """ loop through directories """
            os.chdir(tstep_dir)
            print("--> Entering " + str(os.getcwd()))
            files = []
            if str(os.getcwd()).endswith('_5e5') or str(os.getcwd()).endswith('e6'):  # if timestep is big (>5e5) -> many files 
                files = sorted(glob.glob("my_experiment00*"))   # -> only look at the ones that start with 00
            elif str(os.getcwd()).endswith('_1_5e5'):                                 # tstep = 1e5
                files = sorted(glob.glob("my_experiment0*"))   # -> only look at the ones that start with 0
            else:
                print(str(os.getcwd()) + ', not a special case')
                files = sorted(glob.glob("my_experiment*.csv"))   # -> only look at the ones that start with 0
            file_list = [filename for filename in files if filename.split('.')[1]=='csv']
            file_list = file_list[2:23]  # take only files in this interval (maximum. Can be shorter)
            if len(file_list) == 0:
                continue
            # get the x coordinate values from the first file:

            print("n of files: "+ str(len(file_list)))
            # re-initialise stuff for the DataFrame:
            df = pd.DataFrame()  # every cycle, forget what df was and re-initialise
         # ~~~ start time ~~~
            st1 = time.time()
            # for file in file_list:
            print(file_list)
            x_coordinate_series = read_extract_x_coord(file_list[0])
            with Pool(processes=6) as pool:
                df_list = pool.map(read_extract_P, file_list)
                # reduce the list of dataframes to a single dataframe
                combined_df = pd.concat(df_list)#, ignore_index=True)
                #df.attrs['tstep'] = tstep_dir.split("_")[2]  # get the third part of the name, after _ (eg 5e3 from tstep02_3_5e3)
            # add x_coordinate_series to combined_df
            combined_df = pd.concat([combined_df, x_coordinate_series], axis=1)
            # --- end for loop files ---
            #df_merged = pd.concat(df_list, axis=1, ignore_index=False)  # add columns to the right. ignore_index=False keeps the column name. Keeps 'x_coordinate' from all dfs
            # print("\n\n printing dfs one by one")
            # for df1 in df_list:
            #     print("len(df1): "+str(len(df1))+", df1.columns: "+str(df1.columns))
            #     print(df1)

#             print(df_list[0])
#             df_merged = df_list[0]
# #            df_merged = ft.reduce(lambda left, right: pd.merge(left, right, on='x_coordinate',copy=False), df_list)
#             for d in df_list[1:]: # skip the first one (it was already saved in df_merged in the line above)
#                 df_merged = pd.merge(df_merged, d, on=["x_coordinate"])
#                 # reduce the list of dataframes to a single dataframe



#            df_merged = df_xc.join(df_list, on='x_coordinate')
            print(combined_df)
         # ~~~ end time ~~~
            et1 = time.time()
            elapsed_time1 = et1 - st1
            #print('To build df:', elapsed_time1, 'seconds')
            # at the end of the files loop, I'll have a dataframe with columns: x_coordinate, t_04000, t_08000, etc
            time0_name = time0_dir.split("_")[3]  # from sigma_1_0_gaussTime02 to gaussTime02 
            tstep_name = "t"+tstep_dir.split("_")[2] # from tstep02_3_5e3 to 5e3
            #print(time0_name,tstep_name) # gaussTime02/tstep02_3_5e3

            # append df to a group and assign the key with f-strings
                                                                #f'{group}/{k}'
            combined_df.to_hdf("../../../my_h5_limitFiles0_sigma3_par.h5", f'{sigma_dir}/{time0_name}/{tstep_name}', append=True)
            # !!!! if you change the name here, remember to change it where it checks if it exists  !!!!
            del combined_df,file_list
            os.chdir("..")
            # --- end tstep_dir loop ---
        os.chdir("..")
        # --- end time0_dir loop (e.g. sigma_1_0_gaussTime02) ---
    os.chdir("..")
    # --- end of sigma_dir loop (e.g. sigma_1_0)

# ~~~ end time ~~~
eta = time.time()
elapsed_timea = eta - sta
print('Tot time:', elapsed_timea, 'seconds')