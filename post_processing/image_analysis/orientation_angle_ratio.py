"""
Find peak_locations in orientation data

"""

import numpy as np
import os
import sys
import pandas as pd
import json
import glob

sys.path.append("/home/home01/scgf/myscripts/post_processing")

from viz_functions import find_dirs,find_variab

real_timestep = 24000.0
variab = find_variab()

def get_rose_histogram():
    # Load the JSON data from the file
    with open('segments_angles.json', 'r') as file:
        data = json.load(file)

    # Initialize a variable to store the rose_histogram values
    rose_histogram = None

    # Iterate through the data to find the matching timestep
    for entry in data:
        if entry['timestep_n'] == real_timestep:
            rose_histogram = entry['rose_histogram']
            break

    # Check if we found the values and print them
    if rose_histogram is not None:
        print(f"Rose histogram values for timestep {real_timestep}: {rose_histogram}")
    else:
        print(f"No data found for timestep {real_timestep}")
    
    return rose_histogram

# Function to count the number of peak_locations in the histogram
def count_peak_locations(histogram):
    peak_count = 0
    for i in range(1, len(histogram) - 1):
        if histogram[i] > histogram[i - 1] and histogram[i] > histogram[i + 1]:
            peak_count += 1
    return peak_count


import matplotlib.pyplot as plt

def find_peak_locations(histogram):
    # Identifying the peak_locations
    peak_locations = []
    peaks = []
    for i in range(1, len(histogram) - 1):
        if histogram[i] > histogram[i - 1] and histogram[i] > histogram[i + 1]:
            peak_locations.append(i)
            peaks.append((i, histogram[i]))

    # keep only the first half (it's duplicated)
    peaks = peaks[:len(peaks) // 2]
    
    # Sorting the peaks based on their values (second element of the tuple)
    peaks.sort(key=lambda x: x[1], reverse=True)
    print(f' len(peaks): { len(peaks)}')

    # Extracting the top two peaks
    max_peak = peaks[0] if len(peaks) > 0 else (-1,-1)
    second_max_peak = peaks[1] if len(peaks) > 1 else (-1,-1)  # impossible number that works as a flag for saying "something is wrong"

    print(f'max peak {max_peak}, second {second_max_peak}. Ratio: {max_peak[1]/second_max_peak[1]}')  # the first output is the index
    
    if False:  # in case I want to visualise the rose diagram
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, projection='polar')
        
        print(f'peak_locations: {peak_locations}')
        # print(f'histogram[peak_locations]: {[histogram[p] for p in peak_locations]}')

        hist_peak = np.zeros(len(histogram))


        for p in peak_locations:
            hist_peak[p] = histogram[p]

        
        print(f'hist_peak {hist_peak}')
        

        # the height of each bar is the number of angles in that bin
        # ax.grid(False)
        ax.bar(np.deg2rad(np.arange(0, 360, 10)), histogram, 
            width=np.deg2rad(10), bottom=0.0, color=(0.5, 0.5, 0.5, 0.2), edgecolor='k')
            # width=np.deg2rad(10), bottom=0.0, color=(1, 0, 0), edgecolor='r')

        # Highlighting the peak_locations
        for peak in peak_locations:
            # hist_peak = 
            ax.bar(np.deg2rad(np.arange(0, 360, 10)), hist_peak, 
                width=np.deg2rad(10), bottom=0.0, color=(0.8, 0.0, 0.0, 0.1), edgecolor='k')
            # ax.bar(peak, histogram[peak], color='red')

        ax.set_theta_zero_location('N') # zero starts at North
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.arange(0, 360, 10), labels=np.arange(0, 360, 10))
        ax.set_rgrids(np.arange(1, max(histogram) + 1, 2), angle=0, weight= 'black')

        plt.show()
    # if len(peaks)>1:
    return max_peak[0]*10, second_max_peak[0]*10, max_peak[1]/second_max_peak[1]   # the first value of max_peak identifies the nth bin in the histogram. Multiply it by 10 to get the angle in degrees


x_variables = find_dirs(variab)
orient_list = []


for v in x_variables:
    os.chdir(v) 
    for mr in sorted(glob.glob("vis*_mR_0*")):   # find directories that match this pattern
        os.chdir(mr)
        rose_histogram = []
        if variab == "viscosity":
            x_variable = v.split('_')[2]  # take the third part of the string, the one that comes after _     -> from visc_1_1e1 to 1e1
        elif variab == "def_rate":
            x_variable = v.split('def')[1]  # take the second part of the string, the one that comes after def     -> from pdef1e8 to 1e8
        melt_rate_value = '0.00'+mr.split('mR_0')[1]  # from full name of directory to 0.001, 0.002 etc

        print(os.getcwd())
        df = pd.DataFrame(columns=['time_eq','viscosity','melt_rate', 'angle_between_peaks','ratio'])  # new dataframe with columns
        rose_histogram = get_rose_histogram()

        if rose_histogram is not None and any(value != 0 for value in rose_histogram):  # check if any are not zero

            max_peak_angle, second_max_peak_angle, ratio = find_peak_locations(rose_histogram)  # extract peaks from histogram
            print(max_peak_angle,second_max_peak_angle,ratio)
            print(abs(max_peak_angle-second_max_peak_angle),ratio)
            if max_peak_angle != -10 and second_max_peak_angle != -10:  # avoid the data with the flag "-1" (now it is -10 because it has been multipl by 10 to get the angle)
                # create dictionary with data from this specific simulation
                orientation_data = {
                    'time_eq': real_timestep,
                    'viscosity': x_variable,
                    'melt_rate': melt_rate_value,
                    'angle_between_peaks': abs(max_peak_angle-second_max_peak_angle),
                    'ratio': ratio
                }
                orient_list.append(orientation_data)
        os.chdir('..')
    os.chdir('..')

# create a dataframe from the list
df = pd.DataFrame(orient_list)

print(df)
