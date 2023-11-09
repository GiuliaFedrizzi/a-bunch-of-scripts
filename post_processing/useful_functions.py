import os.path

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

