import os
import pandas as pd
import re   # regex

def find_innermost_directories(root_dir):
    """Recursively find all innermost directories containing CSV files."""
    innermost_dirs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Filter directories without subdirectories and containing CSV files
        if not dirnames and any(fname.endswith('.csv') for fname in filenames):
            innermost_dirs.append(dirpath)
    return innermost_dirs
# Get the number from the filename
def extract_number(filename):
    match = re.search(r"(\d+)", filename)
    if match:
        return int(match.group(1))
    return 0

def extract_min_max_porosity(directories):
    """Extract the minimum and maximum porosity values from the last CSV file in each directory."""
    all_poro_min =[]
    all_poro_max =[]
    for directory in directories:
        csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
        if csv_files:
            # Sort and select the last CSV file
            last_csv_file = sorted(csv_files,key=extract_number)[-1]
            # Read the CSV file
            df = pd.read_csv(os.path.join(directory, last_csv_file))
            # Extract minimum and maximum values
            min_porosity = df['Porosity'].min()
            max_porosity = df['Porosity'].max()
            all_poro_min.append(min_porosity)
            all_poro_max.append(max_porosity)

    return min(all_poro_min),max(all_poro_max)

# Define the root directory of your directory tree
root_dir = os.getcwd()

# Find all innermost directories containing CSV files
innermost_dirs = find_innermost_directories(root_dir)

# Extract the minimum and maximum porosity values
porosity_values = extract_min_max_porosity(innermost_dirs)
print(f'Global porosity range {porosity_values}')
f = open("global_porosity_range.txt", "w")
f.write(str(porosity_values))
f.close()
# # Print or further process the extracted values
# for file_path, min_max_values in porosity_values.items():
#     print(f'{file_path}: Min Porosity = {min_max_values[0]}, Max Porosity = {min_max_values[1]}')
