"""
get the minimum and maximum porosity in all of the simulations in all of the subdirectories
save it to global_porosity_range.txt
"""

import os
import pandas as pd

def find_latest_experiment_file(directory):
    """Find the latest my_experiment*.csv file in the given directory."""
    experiment_files = [f for f in os.listdir(directory) if f.startswith("my_experiment") and f.endswith(".csv")]
    if not experiment_files:
        return None
    latest_file = max(experiment_files, key=lambda x: int(x.split('my_experiment')[1].split('.csv')[0]))
    return os.path.join(directory, latest_file)

def find_global_porosity_range(root_dir):
    min_porosity = float('inf')
    max_porosity = float('-inf')

    for root, dirs, files in os.walk(root_dir):
        for dir in dirs:
            latest_file = find_latest_experiment_file(os.path.join(root, dir))
            if latest_file:
                df = pd.read_csv(latest_file)
                print(f'min {df["Porosity"].min()}, max {df["Porosity"].max()}')
                min_porosity = min(min_porosity, df["Porosity"].min())
                max_porosity = max(max_porosity, df["Porosity"].max())
                print(f'so far, global min {min_porosity}, global max {max_porosity}')

    return min_porosity, max_porosity

def write_global_porosity_range(file_path, min_porosity, max_porosity):
    with open(file_path, 'w') as f:
        f.write(f"{min_porosity}\n{max_porosity}")
        # f.write(f"Maximum Porosity: {max_porosity}\n")

if __name__ == "__main__":
    root_directory = os.getcwd()
    output_file = "global_porosity_range.txt"
    
    min_porosity, max_porosity = find_global_porosity_range(root_directory)
    write_global_porosity_range(output_file, min_porosity, max_porosity)
    
    print(f"Global porosity range has been written to {output_file}")
