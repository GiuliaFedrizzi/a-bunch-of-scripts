import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os
import sys
import numpy as np

sys.path.append("/home/home01/scgf/myscripts/post_processing")
from viz_functions import find_dirs,find_variab
from useful_functions import getParameterFromLatte



def get_bb_slow_and_fast(filename):
    df = pd.read_csv(filename, header=0)  # build the dataframe from the csv file

    df = df[(df["x coord"]>0.02) & (df["x coord"]<0.98)] # remove the sides (2% each side)

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



#reference_tstep = 72000
reference_tsteps = [14000]
reference_tsteps = [i*5 for i in reference_tsteps]  # to use the true time step (not normalised by the amount of melt)
fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(12,7))

mr_contrast_dirs = ['lr41','lr44','lr42','lr45','lr43']
mr_contrast = [1,1.5,2,2.5,3]
frac_list = []


for reference_tstep in reference_tsteps:
    for mrc,lr in enumerate(mr_contrast_dirs):
        print(f'lr {lr}')
        os.chdir(lr)
        # def_dirs = sorted((glob.glob("def*")), key=extract_numeric_value)   # used to be for "rt0.*" (relaxation threshold)
        def_dirs = ['def0e-8','def1e-7','def2e-7','def3e-7','def4e-7','def5e-7']   # set manually
        print(f'def_dirs {def_dirs}')
        for def_dir in def_dirs:
            os.chdir(def_dir)
            x_variable = find_dirs()
            for x in x_variable:
                os.chdir(x)

                # melt_labels = find_dirs()
                melt_labels = glob.glob("*05")
                print(os.getcwd())
                
                for m in melt_labels:
                    os.chdir(m)
                    m_rate = float(getParameterFromLatte("input.txt","melt_increment"))
                    norm_time = reference_tstep/(m_rate*1000)  # normalise time according to how much melt waas created (so we have constant melt addition)
                    filename = "my_experiment"+str(int(norm_time)).zfill(5)+".csv"  # build the file name based on the normalised time (+ zeros)
                    if os.path.isfile(filename):
                        slow_n,fast_n = get_bb_slow_and_fast(filename)
                        mu_f = float(getParameterFromLatte("input.txt","mu_f"))
                        def_rate = float(getParameterFromLatte("input.txt","defRate"))
                        frac_data = {
                            'reference_tstep': reference_tstep,
                            'norm_time': norm_time,
                            'Melt Rate': m_rate,
                            'mu_f': mu_f,
                            'Deformation Rate': def_rate/1e8,
                            'slow_n': slow_n,
                            'fast_n': fast_n,
                            'Melt Rate Contrast': mr_contrast[mrc]
                        }
                        frac_list.append(frac_data)
                        # print(frac_data)
                    # else:
                    #     print(f'no file {os.getcwd()}/{filename}')

                    os.chdir('..')
                os.chdir('..')
            os.chdir('..')
            print("\n")
        os.chdir('..')


    df = pd.DataFrame(frac_list)
    df.drop(df[df.fast_n < 5].index, inplace=True)  # filter out if there aren't enough points
    df["slow/total"] = (df["slow_n"] / (df["fast_n"]+df["slow_n"]))*100
    print(df)

    # split the data between low and high viscosity
    df_high_visc = df[df["mu_f"] == 1000]
    df_low_visc = df[df["mu_f"] == 100]

    # df_high_visc_curr_tstep = df_high_visc[df_high_visc["reference_tstep"]==reference_tstep]
    # print(df_high_visc_curr_tstep)

# plot
if len(df_low_visc) > 0:
    sns.scatterplot(data=df_low_visc,x="Deformation Rate",y="slow/total",hue="Melt Rate Contrast",palette="cividis",ax=ax1) 
    sns.lineplot(data=df_low_visc,x="Deformation Rate",y="slow/total",hue="Melt Rate Contrast",palette="cividis", legend=None,ax=ax1) 
if len(df_high_visc) > 0:
    sns.scatterplot(data=df_high_visc,x="Deformation Rate",y="slow/total",hue="Melt Rate Contrast",palette="cividis",ax=ax2) # style='reference_tstep',
    sns.lineplot(data=df_high_visc,x="Deformation Rate",y="slow/total",hue="Melt Rate Contrast",palette="cividis",legend=None,ax=ax2) 

ax1.set_ylabel('Fractures Outside Fertile Layers, %')#,weight='bold')
ax1.set_xlabel('Deformation Rate')#,weight='bold')
plt.ticklabel_format(axis='both', style='sci')
ax2.set_ylabel('Fractures Outside Fertile Layers, %')#,weight='bold')
ax2.set_xlabel('Deformation Rate (s$^{-1}$)')#,weight='bold')

# ax2.set_xscale('log')

# ax1.grid(True)
# ax2.grid(True)
ax2.grid(linestyle='--',alpha=0.6)
ax1.set_title("Low viscosity")
ax2.set_title("High viscosity")
fig.suptitle("$t_{ref} = $ "+str(reference_tstep)+" (t = "+str(float(reference_tstep)/5)+")")

fig.tight_layout()

plt.savefig("slow_fast_bb_"+str(reference_tstep)+".png",dpi=300)
plt.show()
# slow_n,fast_n = get_bb_slow_and_fast(filename)

# print(f'slow_n {slow_n}, fast_n {fast_n}')
