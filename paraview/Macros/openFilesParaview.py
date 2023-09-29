"""
For every subdirectory, go into the directory and open all csv files in paraview's pipeline.
To be run as a macro or in paraview's python shell.
"""
# trace generated using paraview version 5.6.0
#
# To ensure correct image size when batch processing, please search 
# for and uncomment the line `# renderView*.ViewSize = [*,*]`

#### import the simple module from the paraview
from paraview.simple import *
import glob
import os
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

parentDir = os.getcwd() # current directory, above the subdirectories
dirlist=sorted(glob.glob(parentDir+"/*/"))  # list of directories inside the current directory

if len(dirlist)==0:
    raise OSError('No directories here!')  # checking if there are directories

# loop for every directory
for d in dirlist:
    os.chdir(d)     # enter the directory

    #get the "base" name of the subdirectory (not the full path)
    d_base_name = d.replace(parentDir,"")
    d_base_name = d_base_name.replace("/","")   # get rid of slashes

    # create a new name for objects in the pipeline
    object_name = d.replace("/nobackup/scgf/myExperiments/","")  # delete the first part of the path
    object_name = object_name.replace("/","_")  # change slashes (/) into underscores (_)
    object_name = object_name[:-1]   # drop the last character ( the final "/" ")

    if d_base_name.startswith("base") == False:

        #print(os.getcwd())
        allFiles=[]
        for i in sorted(glob.glob("*.csv")):
            allFiles.append(os.getcwd()+"/"+i)   # create a long list of all csv files with their absolute path
        
        if len(dirlist)==0:
            raise OSError('No files here!')

        print(d_base_name)
        # create a new 'CSV Reader' with same name as subdirectory
        d_base_name = CSVReader(FileName=allFiles)
        # get animation scene
        animationScene1 = GetAnimationScene()

        # update animation scene based on data timesteps
        animationScene1.UpdateAnimationUsingDataTimeSteps()
        

        # rename source object
        RenameSource(object_name, d_base_name)  # first: new name, second: old name

    # we're done in this directory, so go back to parent directory
    os.chdir('..')

#### uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).
