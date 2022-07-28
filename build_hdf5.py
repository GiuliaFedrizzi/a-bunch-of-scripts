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

#parent_directory = "/nobackup/scgf/myExperiments/gaussScale/scale050/"

# some variables for the subplots:

fig, axs = plt.subplots(nrows=1, ncols=1)

#os.chdir(parent_directory)

# initialise stuff
max_P = 0.0
min_P = 0.0



if os.path.exists('my_h5.h5'):
    os.remove('my_h5.h5')

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
         # ~~~ start time ~~~
            st = time.time()
            if str(os.getcwd()).endswith('_5e5') or str(os.getcwd()).endswith('e6'):  # if timestep is big (>5e5) -> many files 
                file_list = sorted(glob.glob("my_experiment00*"))   # -> only look at the ones that start with 00
            elif str(os.getcwd()).endswith('_1_5e5'):                                 # tstep = 1e5
                file_list = sorted(glob.glob("my_experiment0*"))   # -> only look at the ones that start with 0
            else:
                print(str(os.getcwd()) + ', not a special case')
                file_list = sorted(glob.glob("my_experiment*.csv"))   # -> only look at the ones that start with 0
         # ~~~ end time ~~~
            et = time.time()
            elapsed_time = et - st
            print('To get the file list:', elapsed_time, 'seconds')
            print("n of files: "+ str(len(file_list)))
            # re-initialise stuff for the DataFrame:
            df = pd.DataFrame()  # every cycle, forget what df was and re-initialise
            x_col_done = 0       # have I put the column with the x coordinates in the dataframe yet?
         # ~~~ start time ~~~
            st1 = time.time()
            for file in file_list[2:23]:
                """ loop through FILES """
                # if i == 0 or i == 1:
                #     continue
                # if i == 20:
                #     continue

                # go through files, one by one
                myExp = pd.read_csv(file, header=0)
                # find the location in the y coordinate corresponding to the max pressure 
                ymax = myExp.loc[myExp['Pressure'].idxmax(), 'y coord']
                # find the x coordinates that corresponds to the max pressure
                x_array = np.array(myExp.loc[myExp.apply(lambda x: math.isclose(x['y coord'],ymax,rel_tol=1e-3),axis=1),'x coord'])
                # extract the pressure arrays
                pressure_array_x = np.array(myExp.loc[myExp.apply(lambda x: math.isclose(x['y coord'],ymax,rel_tol=1e-3),axis=1),'Pressure'])
                #pressure_array_y = np.array(myExp.loc[myExp.apply(lambda x: math.isclose(x['x coord'],xmax,rel_tol=1e-3),axis=1),'Pressure'])
                col_name = file.replace(".csv","").replace("my_experiment","t_")
                if x_col_done == 0:
                    df["x_coordinate"] = x_array
                    x_col_done = 1   # now I've done it, so don't add it next time
                df[col_name] = pressure_array_x  # store x coordinates

                #df.attrs['tstep'] = tstep_dir.split("_")[2]  # get the third part of the name, after _ (eg 5e3 from tstep02_3_5e3)
            # --- end for loop files ---
         # ~~~ end time ~~~
            et1 = time.time()
            elapsed_time1 = et1 - st1
            print('To build df:', elapsed_time1, 'seconds')
            # at the end of the files loop, I'll have a dataframe with columns: x_coordinate, t_04000, t_08000, etc
            time0_name = time0_dir.split("_")[3]  # from sigma_1_0_gaussTime02 to gaussTime02 
            tstep_name = "t"+tstep_dir.split("_")[2] # from tstep02_3_5e3 to 5e3
            print(time0_name,tstep_name) # gaussTime02/tstep02_3_5e3
         # ~~~ start time ~~~
            st3 = time.time()  # append to h5

            # append df to a group and assign the key with f-strings
                                                                #f'{group}/{k}'
            df.to_hdf("../../../my_h5_limitFiles0_sigma3.h5", f'{sigma_dir}/{time0_name}/{tstep_name}', append=True)
            et3 = time.time()
            elapsed_time3 = et3 - st3
            print('To append to h5:', elapsed_time3, 'seconds')

            os.chdir("..")
            # --- end tstep_dir loop ---
        os.chdir("..")
        # --- end time0_dir loop (e.g. sigma_1_0_gaussTime02) ---
    os.chdir("..")
    # --- end of sigma_dir loop (e.g. sigma_1_0)

