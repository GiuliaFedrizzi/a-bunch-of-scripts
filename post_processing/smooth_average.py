import csv

# Define the CSV file name and the column name
csv_file = "fluidgrid00300.csv"
column_name = "tot_smooth"

# Initialize variables to keep track of the sum and count
total = 0
count = 0

max_smooth=0
# Open the CSV file and read its contents
with open(csv_file, mode="r") as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        # Check if the column exists in the current row
        if column_name in row:

            # Try to convert the column value to a float
            value = float(row[column_name])
            total += value
            count += 1

            if max_smooth<value:
                max_smooth = value

# Calculate the average
if count > 0:
    average = total / count
    print(f"The average of column '{column_name}' in '{csv_file}' is: {average:.2f}. Tot rows: {count}, max = {max_smooth}")
else:
    print(f"No valid data found in column '{column_name}' of '{csv_file}'")
