import re
import matplotlib.pyplot as plt

def extract_milliseconds_from_file(file_path):
    # Regular expression to match lines with milliseconds
    pattern = r"Time taken by FullRelax: (\d+) milliseconds"

    # List to hold extracted millisecond values
    millisecond_values = []

    # Read the file and extract values
    with open(file_path, 'r') as file:
        # line_count = 0
        for line in file:
            # if line_count >= 100:
                break
            match = re.search(pattern, line)
            
            if match:
                millisecond_values.append(int(match.group(1)))
                # line_count+=1
                # print(f'line_count {line_count}')

    return millisecond_values

# Assuming the two files are located at these paths
file_path1 = 'op08/vis1e1_mR_01/latte.log'
file_path2 = 'op09/vis1e1_mR_01/latte.log'

# Extract millisecond values from both files
milliseconds1 = extract_milliseconds_from_file(file_path1)
milliseconds2 = extract_milliseconds_from_file(file_path2)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(milliseconds1, alpha = 0.5, label=file_path1+", serial")
plt.plot(milliseconds2, alpha = 0.5, label=file_path2+", parallel")
plt.xlabel('Instance')
plt.ylabel('Milliseconds')
plt.title('Time taken by FullRelax')
plt.legend()
plt.show()