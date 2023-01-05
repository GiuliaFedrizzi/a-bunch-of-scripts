# -*- coding: utf-8 -*-
#### import the simple module from the paraview
from paraview.simple import *
import glob
import os
import time
import re
# add the path to directory with macros to the sys path
sys.path.append('/home/home01/scgf/myscripts/paraview')  # change if path to my_functions.py is different

from my_functions import * 

def get_file_number(fileName):
    searchResult = re.search("([0-9]{5})",fileName)
    if searchResult is None:
        raise Exception("File name formatting incorrect")
    return int(searchResult.group())


#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# read the path, get the scale from it (size020 = 20 metres)
domain_size,found_scale = get_scale()  # domain_size is the sccale, found_scacle is 0 or 1

# some variables for saving animations
thisDirectory = os.getcwd()

# ======== OPEN FILES ================
# find files
allInputFiles = []
latestFile = None 


sortedInputFiles = sorted(glob.glob("my_experiment*.csv"))
for i in sortedInputFiles:
    fileName = os.getcwd() + "/" +i
    allInputFiles.append(fileName)



allOutputFiles = []
for i in sorted(glob.glob("a_porosity_*.png")):
    allOutputFiles.append(os.getcwd() + "/" +i)
    
numberOfFiles = len(allInputFiles)

if len(allOutputFiles) == 0:
    rangeForAnimation = [0,numberOfFiles-1]# from 0 to n-1, because 0 is included (and the last number too)
else:
    latestOutputFileNumber = get_file_number(allOutputFiles[len(allOutputFiles)-1])
    allInputFiles = allInputFiles[latestOutputFileNumber:]
    rangeForAnimation = [latestOutputFileNumber + 1,numberOfFiles - 1]


# ========== GET THE TIMESTEP ==========
# from input.txt
f_input=open("input.txt",'r')   # open input.txt, where time step is stored
for line_num, line in enumerate(f_input):
    if line_num == 1:
        time_string = line       # the second line contains the value of timestep
        break
time_string_no_Tstep = time_string.replace("Tstep ","")   # remove "Tstep " (before the number)
time_step_num = float(time_string_no_Tstep.replace("\n",""))      # remove "\n"     (after the number)


# ========= GET THE FREQUENCY AT WHICH FILES WERE SAVED  ============
   # from the first two file names
list_of_my_exp = sorted(glob.glob("*.csv"))
for i,my_exp in enumerate(list_of_my_exp):
    num = my_exp.replace("my_experiment","")  # take the file name and get rid of my_experiment 
    num = num.replace(".csv","")              # and .csv
    if i == 0:
        first = float(num)
    elif i == 1:
        second = float(num)
        break
frequency=second-first
print("frequency when files were saved: " + str(frequency))

# =========== PARAVIEW STUFF ========

# create a new 'CSV Reader'
my_experiment = CSVReader(FileName=allInputFiles)

# get animation scene
animationScene1 = GetAnimationScene()

# update animation scene based on data timesteps
animationScene1.UpdateAnimationUsingDataTimeSteps()

# get layout
layout1 = GetLayoutByName("Layout #1")

# find view
renderView1 = FindViewOrCreate('RenderView1', viewtype='RenderView')

# set active view
SetActiveView(renderView1)

# rename source object
RenameSource('mf0001_freq1e1', my_experiment)

LoadPalette(paletteName='WhiteBackground')

applyTableToPointsSingle()

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')

applyLightingAnd2Dview()

# update the view to ensure updated data information
renderView1.Update()

# create a new 'Delaunay 2D'
delaunay2D1 = Delaunay2D()

# update the view to ensure updated data information
renderView1.Update()

# create a new 'Transform'
transform1 = Transform(Input=delaunay2D1)
transform1.Transform = 'Transform'

if found_scale:
    # apply the right numbers to transform
    transform1.Transform.Translate = [50.0-(domain_size/2), 100.0-domain_size, 0.0]  # works if scale020,scale030 etc.
    transform1.Transform.Scale = [domain_size, domain_size, domain_size]
else:
  print("No scale")



# show data in view
transform1Display = Show(transform1, renderView1)

# hide data in view
Hide(delaunay2D1, renderView1)

# update the view to ensure updated data information
renderView1.Update()

# === annotate time filter ===
annotateTimeFilter1 = AnnotateTimeFilter(registrationName='AnnotateTimeFilter1', Input=transform1)
annotateTimeFilter1Display = Show(annotateTimeFilter1, renderView1)#, 'TextSourceRepresentation')
annotateTimeFilter1.Format = 'Time (s): %9.2e'
# .........................
if len(allOutputFiles) == 0:
    annotateTimeFilter1.Shift = 0   # first timestep
else:
    annotateTimeFilter1.Shift = (latestOutputFileNumber + 1)*frequency  # if there are already pngs, shift the time
annotateTimeFilter1.Scale = (time_step_num*frequency)   # time between timesteps, extracted from input.txt, times the frequency for saving files
# location of filter
annotateTimeFilter1Display.WindowLocation = 'AnyLocation'

renderView1.Update()

# ===  PRESSURE  ===
set_colormap_pressure(transform1Display)
# save animation
pathAndName = thisDirectory+'/a_f_pressure.png'
SaveAnimation(pathAndName, renderView1, ImageResolution=[2054, 1248],
    FrameWindow=rangeForAnimation, 
    # PNG options
    SuffixFormat='_%05d')
pressureLUT = GetColorTransferFunction('Pressure')

# ===  BROKEN BONDS  ===
set_colormap_broken_bonds(transform1Display)
brokenBondsLUT = GetColorTransferFunction('BrokenBonds')
HideScalarBarIfNotNeeded(pressureLUT, renderView1)
# save animation
pathAndName = thisDirectory+'/a_brokenBonds.png'
SaveAnimation(pathAndName, renderView1, ImageResolution=[2054, 1248],
    FrameWindow=rangeForAnimation, 
    # PNG options
    SuffixFormat='_%05d')

# ===  POROSITY  ===
set_colormap_porosity(transform1Display)
HideScalarBarIfNotNeeded(brokenBondsLUT, renderView1)
# save animation
pathAndName = thisDirectory+'/a_porosity.png'
SaveAnimation(pathAndName, renderView1, ImageResolution=[2054, 1248],
    FrameWindow=rangeForAnimation, 
    # PNG options
    SuffixFormat='_%05d')


# toggle 3D widget visibility (only when running from the GUI)
Hide3DWidgets(proxy=transform1.Transform)



# update the view to ensure updated data information
renderView1.Update()
