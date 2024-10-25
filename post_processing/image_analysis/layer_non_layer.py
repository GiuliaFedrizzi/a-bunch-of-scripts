import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os
import sys

sys.path.append("/home/home01/scgf/myscripts/post_processing")
from viz_functions import find_dirs,find_variab
from useful_functions import getParameterFromLatte


filename = "my_experiment09000.csv"

def get_bb_slow_and_fast(filename):
    df = pd.read_csv(filename, header=0)  # build the dataframe from the csv file

    df = df[(df["x coord"]>0.02) & (df["x coord"]<0.98)]#["Broken Bonds"]

    bb_df_1 = df[(df["y coord"]>0.0) & (df["y coord"]<0.1428) & (df["Broken Bonds"]>0) ]
    bb_df_2 = df[(df["y coord"]>0.1428) & (df["y coord"]<0.2857)& (df["Broken Bonds"]>0) ]
    bb_df_3 = df[(df["y coord"]>0.2857) & (df["y coord"]<0.428) & (df["Broken Bonds"]>0) ]
    bb_df_4 = df[(df["y coord"]>0.428) & (df["y coord"]<0.57) & (df["Broken Bonds"]>0) ]
    bb_df_5 = df[(df["y coord"]>0.57) & (df["y coord"]<0.714) & (df["Broken Bonds"]>0) ]
    bb_df_6 = df[(df["y coord"]>0.714) & (df["y coord"]<0.857) & (df["Broken Bonds"]>0) ]
    bb_df_7 = df[(df["y coord"]>0.857) & (df["y coord"]<0.98) & (df["Broken Bonds"]>0) ]

    bb_slow = len(bb_df_1) + len(bb_df_3) + len(bb_df_5) + len(bb_df_7)
    bb_fast = len(bb_df_2) + len(bb_df_4) + len(bb_df_6)
    
    return bb_slow,bb_fast


def extract_numeric_value(directory_name):
    return float(directory_name[3:])


frac_list = []

def_dirs = sorted((glob.glob("def*")), key=extract_numeric_value)   # used to be for "rt0.*" (relaxation threshold)
print(f'def_dirs {def_dirs}')



for def_dir in def_dirs:
    os.chdir(def_dir)
    x_variable = find_dirs()
    for x in x_variable:
        os.chdir(x)

        melt_labels = find_dirs()
        
        for m in melt_labels:
            os.chdir(m)
            if os.path.isfile(filename):
                slow_n,fast_n = get_bb_slow_and_fast(filename)
                m_rate = float(getParameterFromLatte("input.txt","melt_increment"))
                mu_f = float(getParameterFromLatte("input.txt","mu_f"))
                def_rate = float(getParameterFromLatte("input.txt","defRate"))
                frac_data = {
                    'm_rate': m_rate,
                    'mu_f': mu_f,
                    'def_rate': def_rate,
                    'slow_n': slow_n,
                    'fast_n': fast_n
                }
                frac_list.append(frac_data)
                print(frac_data)

            os.chdir('..')
        os.chdir('..')
    os.chdir('..')
    print("\n")


df = pd.DataFrame(frac_list)

print(df)
# slow_n,fast_n = get_bb_slow_and_fast(filename)

# print(f'slow_n {slow_n}, fast_n {fast_n}')
