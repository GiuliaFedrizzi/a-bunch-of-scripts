import os.path
import re
import glob
import pandas as pd
import numpy as np

def getTimeStep(inputFile):
    with open(inputFile) as iFile:
        for num, line in enumerate(iFile,1):
            if "Tstep" in line:
                input_tstep = line.split(" ")[1]  # split before and after space, take the second word (value of timestep)
                return input_tstep  


def getViscosity(inputFile):
    with open(inputFile) as iFile:
        for num, line in enumerate(iFile,1):
            if "mu_f" in line:
                input_mu_f = line.split(" ")[1]  # split before and after space, take the second word (value of timestep)
                return input_mu_f  
            
def getDepth(inputFile):
    with open(inputFile) as iFile:
        for num, line in enumerate(iFile,1):
            if "depth" in line:
                depth = line.split(" ")[1]  # split before and after space, take the second word (value of timestep)
                return depth 

def getParameterFromInput(inputFile,keyword):
    try:
        with open(inputFile, 'r') as file:
            for line in file:
                if line.startswith(keyword):
                    # Split the line into words and return the word after the keyword
                    parts = line.split()
                    if len(parts) > 1:
                        return float(parts[1])  # Convert the string to a float
    except FileNotFoundError:
        return "File not found."
    except ValueError:
        return "Invalid number format."


def getParameterFromLatte(inputFile,word_to_find):
    """ Function that finds the next string after the specified word (word_to_find) in inputFile.
        Output will be a string, might need converting
    """
    with open(inputFile) as iFile:
        for num, line in enumerate(iFile,1):
            if word_to_find in line:
                word_found = line.split(" ")[1]  # split before and after space, take the second word (value of variable)
                return word_found   # stop searching, return the value (will be a string)
def getDensity():
    exp_file = "experiment.cc"
    if not os.path.isfile(exp_file):
        exp_file = "../baseFiles/experiment.cc"  # sometimes it is only stored in the baseFiles directory
        with open(exp_file) as iFile:
            for num, line in enumerate(iFile,1):
                if "setSolidDensity" in line:
                    # take the second part - after ( - then take the first part - before )
                    word_found = (line.split("(")[1]).split(")")[0]  
                    print(f'density  {word_found}')
                    return float(word_found)   # stop searching, return the value (will be a string)

def getWhenItRelaxed(lattefile):
    """ when you find "first complete relaxation", return the first next file that was saved """
    found_compl_relax = False;
    relaxed_file = " "
    with open(lattefile) as expFile:
        for num, line in enumerate(expFile,1):
            if 'first complete relaxation' in line:  # search for the string
                found_compl_relax = True    # ok, it has relaxed
            if found_compl_relax:           # so it can go on and search for "relax times"
                if "Saving file:" in line:
                    relaxed_file = line.split(": ")[1]   # this line look like: Saving file: my_experiment00000.csv
                       # take the last part splitting around ": ", so something like "my_experiment00000.csv"
    return relaxed_file 


def getFirstBrokenBond(lattefile):
    broken_bond_string = "Broken Bonds"
    time_string = "time is "
    with open(lattefile) as expFile:
        for num, line in enumerate(expFile,1):
            # if num <1000:  # Skip the first x lines
            #     continue
            if time_string in line:
                time_line_whole = line
            if broken_bond_string in line:  # at the FIRST occurrence of Broken Bonds, stop the search
                x = time_line_whole.split("time is ") # split the line with the info about time around the string "time is"
                timebb = x[1]    # take the second part, which is the one that comes after the string "time is"
                timebb = timebb.replace("\n","")  # get rid of \n
                return timebb  

def getResolution():
    def extract_number_from_files():
        # List all files that start with "sub"
        files = glob.glob("sub*.sh") + glob.glob("../sub*.sh")  # it could be one level of directories up
        for file_path in files:
            try:
                with open(file_path, 'r') as file:
                    content = file.read()
                    # Regular expression to find "res" followed by any number of digits and then ".elle"
                    match = re.search(r'res(\d+)\.elle', content)
                    if match:
                        return match.group(1), file_path  # Return the number and the file where it was found
            except FileNotFoundError:
                continue  # If file not found, just go to the next file
        
        return "No matching pattern found in any file.", None    

    number, found_file = extract_number_from_files()
    if found_file:
        print(f"Number found: {number} in file: {found_file}")
        return int(number)
    else:
        print(number)
        return 200

def getSaveFreq():
    def extract_number_from_files():
        # List all files that start with "sub"
        files = glob.glob("sub*.sh")
        for file_path in files:
            try:
                with open(file_path, 'r') as file:
                    content = file.read()
                    # Regular expression to find "res" followed by any number of digits and then ".elle"
                    match = re.search(r'saveCsvInterval=(\d+)', content)
                    if match:
                        return match.group(1), file_path  # Return the number and the file where it was found
            except FileNotFoundError:
                continue  # If file not found, just go to the next file
        
        return "No matching pattern found in any file.", None    

    number, found_file = extract_number_from_files()
    if found_file:
        print(f"Number found: {number} in file: {found_file}")
        return number
    else:
        print(number)
        return 100  # just a default


def extract_two_profiles(filename, var_to_plot,point_indexes,res):
    """ open file (filename), extract two profiles of "var_to_plot" (e.g. porosity), returns their coordinates (x_v or x_h) and their values (variable_vals_v or variable_vals_h) """
    if os.path.isfile(filename):
    
        # res = getResolution()
        myExp = pd.read_csv(filename, header=0)
        # df_v = myExp[50:len(myExp):res]    # dataframe only containing a vertical line. start from the 50th element and skip 2 rows of 200 elements
        df_v = myExp[int(point_indexes[1]):len(myExp):int(2*res)]    # dataframe only containing a vertical line. start from the n-th element and skip 2 rows of 200 elements
        hor_index_start = int(point_indexes[0])  # index for the horizontal profile (constant y)
        df_h = myExp[hor_index_start:hor_index_start+res:1]    # dataframe only containing a horizontal line. start from the left boundary, continue for 200 elements
        variable_vals_v = df_v[var_to_plot].values
        variable_vals_h = df_h[var_to_plot].values
        x_v = np.arange(0,1,1/len(df_v))
        x_h = np.arange(0,1,1/len(df_h))
        return (x_v, variable_vals_v), (x_h, variable_vals_h)
    else:
        print("Problem! No file called "+str(filename)+"!")

def extract_horiz_sum_of_bb(filename,res):
    if os.path.isfile(filename):
        myExp = pd.read_csv(filename, header=0)
        broken_bonds = np.zeros(len(myExp)+1) # add an initial one because the bottom left corner is missing
        broken_bonds[1:]=(myExp["Broken Bonds"].values)  # just bb values
        grid_height = (len(myExp)+1) // res
        grid = broken_bonds.reshape(grid_height, res)  # grid_height is the number of rows, 'res' the n of columns
        bb_sum =  grid.sum(axis=1)  # sum along each row
        y_coords = np.linspace(0, 1, grid_height)
        print(f'max bb_sum {max(bb_sum)}')
        return bb_sum,y_coords 
    else:
        print("Problem! No file called "+str(filename)+"!")



def average_flow_ratio(filename,ver_points,hor_points,vars_to_plot):
    int_hor_for_average = []
    int_ver_for_average = []
    for ver_point in ver_points:
        for hor_point in hor_points:
            point_indexes = [hor_point,ver_point]

            # prepare to store vertical and horizontal data
            all_data_v = {}
            all_data_h = {}

            for v in vars_to_plot:
                (x_v, y_v), (x_h, y_h) = extract_two_profiles(filename, v,point_indexes)
                all_data_v[v] = (x_v, y_v)
                all_data_h[v] = (x_h, y_h)
                

            x_coord_vel_hor, y_vel_hor = all_data_h[vars_to_plot[1]] 
            x, poro_values_hor = all_data_h[vars_to_plot[2]]


            integral_hor = np.trapz((y_vel_hor), x=x_coord_vel_hor)
            print(f'integral_hor {integral_hor}')
            int_hor_for_average.append(integral_hor)
            
        y_coord_vel_ver, x_vel_ver = all_data_v[vars_to_plot[0]] 
        x_vel_ver = -x_vel_ver
        x, poro_values_ver = all_data_v[vars_to_plot[2]]
        integral_ver =  np.trapz((x_vel_ver*poro_values_ver), x = y_coord_vel_ver)
        print(f'integral_ver {integral_ver}')
        int_ver_for_average.append(integral_ver)
    average_ver = sum(int_ver_for_average)/len(int_ver_for_average)
    average_hor = sum(int_hor_for_average)/len(int_hor_for_average)
    print(f'average_hor {average_hor}')
    print(f'average_ver {average_ver}')
    # int_ratio = average_hor/abs(average_ver)
    int_ratio = average_hor/average_ver

    return average_hor,average_ver,int_ratio

