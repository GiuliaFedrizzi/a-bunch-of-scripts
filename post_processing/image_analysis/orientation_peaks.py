"""
Find peak_locations in orientation data

"""

import numpy as np

# Provided rose_histogram data
rose_histogram = [0.011235955056179775, 0.0, 0.0, 0.018726591760299626, 0.052434456928838954, 
                  0.27715355805243447, 1.5131086142322094, 0.27340823970037453, 0.0, 1.3707865168539324, 
                  1.6816479400749063, 0.7640449438202248, 1.3370786516853936, 0.3707865168539328, 
                  0.029962546816479474, 0.011235955056179137, 0.0, 0.0, 0.011235955056179775, 0.0, 
                  0.0, 0.018726591760299626, 0.052434456928838954, 0.27715355805243447, 1.5131086142322094, 
                  0.27340823970037453, 0.0, 1.3707865168539324, 1.6816479400749063, 0.7640449438202248, 
                  1.3370786516853936, 0.3707865168539328, 0.029962546816479474, 0.011235955056179137, 
                  0.0, 0.0]

# Function to count the number of peak_locations in the histogram
def count_peak_locations(histogram):
    peak_count = 0
    for i in range(1, len(histogram) - 1):
        if histogram[i] > histogram[i - 1] and histogram[i] > histogram[i + 1]:
            peak_count += 1
    return peak_count

# Counting peak_locations
number_of_peak_locations = count_peak_locations(rose_histogram)
print(number_of_peak_locations)

import matplotlib.pyplot as plt

# Reusing the provided rose_histogram data and the count_peak_locations function

def plot_histogram_with_peak_locations(histogram):
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

    # Extracting the top two peaks
    max_peak = peaks[0] if len(peaks) > 0 else None
    second_max_peak = peaks[1] if len(peaks) > 1 else None

    print(f'max peak {max_peak}, second {second_max_peak}. Ratio: {max_peak[1]/second_max_peak[1]}')
    
    # # Plotting the histogram
    # plt.figure(figsize=(10, 6))
    # plt.bar(range(len(histogram)), histogram, color='gray')

    # # Highlighting the peak_locations
    # for peak in peak_locations:
    #     plt.bar(peak, histogram[peak], color='red')

    # # plt.title('Rose Histogram with peak_locations Highlighted')
    # plt.xlabel('Index')
    # plt.ylabel('Value')
    
    print(f'peak_locations: {peak_locations}')
    # print(f'histogram[peak_locations]: {[histogram[p] for p in peak_locations]}')

    hist_peak = np.zeros(len(histogram))


    for p in peak_locations:
        hist_peak[p] = histogram[p]

    
    print(f'hist_peak {hist_peak}')
    
    fig = plt.figure(figsize=(8,8))

    ax = fig.add_subplot(111, projection='polar')
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

    print(f'max: {max}')

    plt.show()

plot_histogram_with_peak_locations(rose_histogram)



