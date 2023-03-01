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
    print(word_to_find)
    with open(inputFile) as iFile:
        for num, line in enumerate(iFile,1):
            if word_to_find in line:
                print(line)
                word_found = line.split(" ")[1]  # split before and after space, take the second word (value of timestep)
                return word_found   # stop searching, return the value (will be a string)

