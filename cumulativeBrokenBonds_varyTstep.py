#!/home/home01/scgf/.conda/envs/condaPandas/bin/python3

""" plots the cumulative number of broken bonds with increasing time
 Giulia June 2022
 To be run from gaussTime or similar (just above sigma_*_*), goes through sigma*/sigma*_gaussTime05/tstep04_5_1e5
 the last directory (e.g. tstep04_5_1e5) can be given as a command line argument

 keep sigma the same, plot different timesteps on same plot
"""
# goes through multiple directories
import matplotlib.pyplot as plt
import glob
import os as os
import sys

#timestep_dir = "tstep04_3_5e3"
#sigma_dir = sys.argv[1]  # give the path of the last directory (e.g. tstep04_3_5e3). 
# tstep_dir_list = ["tstep04_3_1e3", "tstep04_3_5e3", "tstep04_4_1e4", "tstep04_4_5e4", "tstep04_5_1e5", "tstep04_5_5e5", "tstep04_6_1e6", "tstep04_6_5e6"]
tstep_dir_list = ["tstep_1_1e1", "tstep_2_1e2", "tstep_3_1e3", "tstep_4_1e4"]

fig, axs = plt.subplots(nrows=1, ncols=1)
broken_bond_string = "Broken Bonds"
time_string = "time is "
# get the scale from the path
if os.getcwd().split("/")[-1] == "scale050":  # if I'm in the directory "scale050"
    scale_string = "050"
elif os.getcwd().split("/")[-1] == "scale100":
    scale_string = "100"
elif os.getcwd().split("/")[-1] == "gaussTime":  # gaussTime's scale was 200
    scale_string = "200"
else:
    scale_string = "unknown"

def getBrokenBondTime(lattefile):
    """ every time you find "Broken Bonds", save the number of bb present and the relative time"""
    time_line = 0
    time_bb_array = [];bb_array = []  # reset these two arrays
    with open(lattefile) as expFile:
        for num, line in enumerate(expFile,1):
            if time_string in line:
                time_line = num  # gets overwritten every time until I find a broken bond. Then the value stays
                time_line_whole = line   # save the whole line where you found  "time is "
            if broken_bond_string in line:  # every time you find "Broken Bonds", save the number of bb present and the relative time
                x = time_line_whole.split("time is ") # split the line with the info about time around the string "time is"
                timebb = x[1]    # take the second part, which is the one that comes after the string "time is"
                timebb = timebb.replace("\n","")  # get rid of \n
                broken_bond_number = line.split("Broken Bonds ")[1]  # take what comes after "Broken Bonds"
                broken_bond_number = broken_bond_number.split(",")[0]   # take what comes before the space and comma
                bb_array.append(float(broken_bond_number))  # build the array with the number of bb present at the time
                time_bb_array.append(float(timebb))              # build the array with the relative time 
    return time_bb_array,bb_array  
    
def plot_bb_in_time(timestep_dir):
    # change directory (I want to have 1 script that I keep outside of the many directories)
    os.chdir(timestep_dir)
    #os.chdir(timestep_dir)  # get inside tstep04_... (chooses always the same timestep)
    print(os.getcwd())
    #try:
    time_bb_array,bb_array = getBrokenBondTime("latte.log")
    #except:
    #    print("Failed to get bb or time.")

    os.chdir('..')
    #my_label = "$\sigma$ = " + parent_directory.split("_")[1] + "." + parent_directory.split("_")[2].replace("/","")  # build the label sigma = x (from the directory name)
    my_label = "time step = " + timestep_dir.split("_")[2]  # build the label "time step" (from the directory name)
    axs.plot(time_bb_array,bb_array, '--o',linewidth=1,markersize=2,label=my_label)  # ...and plot it
    #axs.set_xscale('log')
    #axs.set_yscale('log')
    #os.chdir("../..")

#os.chdir(sigma_dir)
#os.chdir(sigma_dir.replace("/","") + "_gaussTime05")  # build the path
for timestep_dir in tstep_dir_list:   # tstep_*
    plot_bb_in_time(timestep_dir)

fig_title = "Number of broken bonds (cumulative) in time for different time steps, \n"+str(os.getcwd())#$\sigma$ = " + sigma_dir.split("_")[1] + "." + sigma_dir.split("_")[2].replace("/","") + "\nscale = "+scale_string
axs.set(title=fig_title, xlabel='Time (s)', ylabel='Number of broken bonds',ylim=[0,2000])
axs.grid(linestyle='--',alpha=0.6)#(linestyle='-', linewidth=2)
axs.set_xscale('log')
# #figure_name = parent_directory.replace("/","-")+"_time_first_bb.png"
# #plt.savefig(figure_name, dpi=600)
plt.legend()
plt.show()

