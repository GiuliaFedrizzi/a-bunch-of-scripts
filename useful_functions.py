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