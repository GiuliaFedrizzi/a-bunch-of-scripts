#!/home/home01/scgf/.conda/envs/condaPandas/bin/python3

""" plots the number of times relaxation was done at each timestep
 Giulia Dec 2022
version for gaussShearFixed, where the 'inner' directories are called 'size020', 'size030', etc
"""
# goes through multiple directories
import matplotlib.pyplot as plt
import os as os
import numpy as np


dir_labels = ['020','030', '040','050','060','070','080','090','100']
resolution = 200
cmap = plt.get_cmap("tab10")  # default colormap (blue,orange,green etc)
plot_num = 0    # a counter for the number of lines. It's used to assign the same colour to lines of the same directory (one real, one from the linear fit)

parent_directories = []
first_part_of_path = '/nobackup/scgf/myExperiments/gaussScaleFixFrac2/g2_13_rad_wGrav200/size'

for i in dir_labels:
    parent_directories.append(first_part_of_path+str(i))

fig, axs = plt.subplots(nrows=1, ncols=1)

relax_string = "relax times"
time_string = "time is "

# get the scale from the path
scale_string = os.getcwd().split("/")[-1]

def getRelaxAndTime(lattefile):
    """ every time you find "relax times", save the number after the string and the relative time"""
    time_line = 0
    found_t0 = False;
    relax_times_array = [];relax_array = []  # reset these two arrays
    with open(lattefile) as expFile:
        for num, line in enumerate(expFile,1):
            if 'time step:0' in line:  # search for the 0th timestep. It shouldn't search for "relax times" until the code starts with timestep 0 
                found_t0 = True    # ok, it found the first timestep
            if found_t0:           # so it can go on and search for "relax times"
                if time_string in line:
                    time_line = num  # gets overwritten every time until I find "relax times". Then the value stays
                    time_line_whole = line   # save the whole line where you found  "time is "
                
                if relax_string in line:  # every time you find "relax times", save the number and the relative time
                    x = time_line_whole.split("time is ") # split the line with the info about time around the string "time is"
                    time_relax_times = x[1]    # take the second part, which is the one that comes after the string "time is"
                    time_relax_times = time_relax_times.replace("\n","")  # get rid of \n
                    relax_times = line.replace("relax times: ","")  # take what comes after "relax times"
                    relax_times = relax_times.split(",")[0]   # take what comes before the space and comma
                    if float(relax_times) > 1:
                        relax_array.append(float(relax_times))  # build the array with the number of relax times at the time
                        relax_times_array.append(float(time_relax_times))              # build the array with the relative time 
                    if float(time_relax_times) == 5e6:
                        break
    return relax_times_array,relax_array  
    
def plot_relax_v_time(parent_directory,plot_num):
    """plot the number of times relaxation was done each timestep"""
    time_array,rel_array = getRelaxAndTime(parent_directory+"/latte.log")
    domain_size=parent_directory.split("/")[-1].replace("size","")   # get the size from the path (name of directory)
    my_label = "size = " + domain_size
    axs.plot(time_array,rel_array, 'o',linewidth=1,markersize=0.8,label=my_label,color=cmap(plot_num),alpha=0.6)  # ...and plot it
    axs.set(ylabel='Relaxation times')
    # fit a line
    m2,q2 = np.polyfit(time_array,rel_array,1) #  fit a line, get m and q
    predict2 = [m2*i+q2 for i in time_array]  # save the array with predicted values 
    plt.plot(time_array,predict2,color=cmap(plot_num))
    plot_num+=1
    return plot_num
    # color=

for parent_directory in parent_directories:
    plot_num = plot_relax_v_time(parent_directory,plot_num)

fig_title = "Relaxation times\n " + first_part_of_path + "*"

axs.set(title=fig_title, xlabel='Time (s)')
axs.grid(linestyle='--',alpha=0.6)#(linestyle='-', linewidth=2)
# #axs.set_xscale('log')
# #figure_name = parent_directory.replace("/","-")+"_times_relax.png"
# #plt.savefig(figure_name, dpi=600)
# plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.legend()
plt.show()

