import re
import matplotlib.pyplot as plt
import os
from itertools import accumulate

plt.figure(figsize=(10, 6))
data_to_plot = []
def extract_milliseconds_from_file(data_to_plot,file_path,function_to_track):
    # Regular expression to match lines with milliseconds
    pattern = r"Time taken by "+function_to_track+": (\d+) milliseconds" 
    # print(pattern)
    # List to hold extracted millisecond values
    minutes_values = []

    # Read the file and extract values
    with open(file_path, 'r') as file:
        line_count = 0
        for line in file:
            if line_count >= 1000:
                print("stop")
                break
            match = re.search(pattern, line)
            
            if match:
                minutes_values.append(int(match.group(1))/1000/60) # from milliseconds to s to min
                line_count+=1
                # print(f'line_count {line_count}')
                # Store the data and label in a dictionary
                data_dict = {
                    'label': file_path,
                    'data': minutes_values
                }

        # Append the dictionary to the list
        data_to_plot.append(data_dict)     
        # print(f'data_to_plot[-1]: {data_to_plot[-1]}') 

    return data_to_plot

# function that we're tracking:
function_to_track = "FullRelax"  #   Run FullRelax

ns = [d for d in os.listdir('.') if os.path.isdir(d) and d.isdigit()]
ns = sorted(ns, key=lambda x: int(x))
for n in ns:
    print("dir is "+n)
    # paths to files:

    file_path = n+'/vis1e1_mR01/latte.log'
    if not os.path.exists(file_path):
        break

    # Extract millisecond values from both files
    data_to_plot = extract_milliseconds_from_file(data_to_plot,file_path,function_to_track)

    # Plotting
    # plt.plot(data_to_plot, alpha = 0.5, label=file_path)
# print(data_to_plot)

data_cumul = data_to_plot

for data_dict in data_cumul:
    data_dict['data'] = list(accumulate(data_dict['data']))

# for data in data_to_plot:
for data in data_cumul:
    # data = [x / 1000 for x in data]
    plt.plot(data['data'], alpha=0.5, label=data['label'])

plt.xlabel('Timestep')
plt.ylabel('Minutes')
plt.title('Time taken by '+function_to_track)
plt.grid(True)
plt.legend()
plt.show()