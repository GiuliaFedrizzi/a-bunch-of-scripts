# -*- coding: utf-8 -*-
#### import the simple module from the paraview
from paraview.simple import *
import glob
import os
import time
import re


#  to sort files correctly:
#  https://stackoverflow.com/a/62941534
import math
# from pathlib import Path 

# file_pattern = re.compile(r'.*?(\d+).*?')
def get_order(file):
    file_number = int(re.findall(r'\d+',file)[0])
    return file_number



sys.path.append(os.getcwd())  # so that it looks for 'modules' (paravParam.py) in the current directory
import paravParam

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
allInputCsvFiles = []
latestFile = None 

sortedInputFiles = sorted(glob.glob('my_experiment*.csv'), key=get_order)   # order files based on their full number, found with get_order()
print(allInputCsvFiles)
#sortedInputFiles = sorted(glob.glob("my_experiment*.csv"))   # old: gets order wrong
for i in sortedInputFiles:
    fileName = os.getcwd() + "/" +i
    allInputCsvFiles.append(fileName)  

allOutputPngFiles = []
for i in sorted(glob.glob("a_porosity_*.png")):
    allOutputPngFiles.append(os.getcwd() + "/" +i)
    
numberOfInputCsvFiles = len(allInputCsvFiles)
print("numberOfInputCsvFiles ",str(numberOfInputCsvFiles))
print("len(allOutputPngFiles) ",str(len(allOutputPngFiles)))

if numberOfInputCsvFiles > 301:
    numberOfInputCsvFiles = 301   # if too many files, stop earlier

rangeForAnimation = [0,numberOfInputCsvFiles-1]# from 0 to n-1, because 0 is included (and the last number too)
latestOutputPngFileNumber = 0

# if len(allOutputPngFiles) == 0:
#     rangeForAnimation = [0,numberOfInputCsvFiles-1]# from 0 to n-1, because 0 is included (and the last number too)
#     latestOutputPngFileNumber = 0
# else:
#     latestOutputPngFileNumber = get_file_number(allOutputPngFiles[len(allOutputPngFiles)-1])
#     allInputCsvFiles = allInputCsvFiles[latestOutputPngFileNumber:]
#     rangeForAnimation = [latestOutputPngFileNumber + 1,numberOfInputCsvFiles - 1]


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
list_of_my_exp = sorted(glob.glob("my_experiment*.csv"))
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
my_experiment = CSVReader(FileName=allInputCsvFiles)

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

# LoadPalette(paletteName='WhiteBackground')

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
# if len(allOutputPngFiles) == 0:
#     annotateTimeFilter1.Shift = 0   # first timestep
# else:
#     annotateTimeFilter1.Shift = (latestOutputPngFileNumber + 1)*frequency  # if there are already pngs, shift the time
annotateTimeFilter1.Shift = (latestOutputPngFileNumber + 1)*frequency  # if there are already pngs, shift the time
print("latestOutputPngFileNumber = ",str(latestOutputPngFileNumber)," shift = ",str((latestOutputPngFileNumber + 1)*frequency))
annotateTimeFilter1.Scale = (time_step_num*frequency)   # time between timesteps, extracted from input.txt, times the frequency for saving files
# location of filter
annotateTimeFilter1Display.WindowLocation = 'AnyLocation'
annotateTimeFilter1Display.FontSize = 120

renderView1.Update()

# ===  PRESSURE  ===
if paravParam.pressure:
    try:
        set_colormap_pressure(transform1Display)
        # save animation
        pathAndName = thisDirectory+'/a_f_pressure.png'
        SaveAnimation(pathAndName, renderView1, ImageResolution=[2054, 1248],
            FrameWindow=rangeForAnimation, 
            # PNG options
            SuffixFormat='_%05d',FontScaling='Do not scale fonts')
        pressureLUT = GetColorTransferFunction('Pressure')
    except:
        pass

# ===  BROKEN BONDS  ===
if paravParam.broken_bonds:
    try:
        set_colormap_broken_bonds(transform1Display)
        brokenBondsLUT = GetColorTransferFunction('BrokenBonds')
        # HideScalarBarIfNotNeeded(pressureLUT, renderView1)
        # save animation
        pathAndName = thisDirectory+'/a_brokenBonds.png'
        SaveAnimation(pathAndName, renderView1, ImageResolution=[2054, 1248],
            FrameWindow=rangeForAnimation, 
            # PNG options
            SuffixFormat='_%05d',FontScaling='Do not scale fonts')
    except:
        pass

# ===  POROSITY  ===
if paravParam.porosity:
    try:
        set_colormap_porosity(transform1Display)
        #HideScalarBarIfNotNeeded(brokenBondsLUT, renderView1)
        # save animation
        pathAndName = thisDirectory+'/a_porosity.png'
        SaveAnimation(pathAndName, renderView1, ImageResolution=[2054, 1248],
            FrameWindow=rangeForAnimation, 
            # PNG options
            SuffixFormat='_%05d',FontScaling='Do not scale fonts')
    except:
        pass


if paravParam.mean_stress:
    try:
        set_colormap_mean_stress(transform1Display)
        # HideScalarBarIfNotNeeded(brokenBondsLUT, renderView1)
        # save animation
        pathAndName = thisDirectory+'/a_meanStress.png'
        SaveAnimation(pathAndName, renderView1, ImageResolution=[2054, 1248],
            FrameWindow=rangeForAnimation, 
            # PNG options
            SuffixFormat='_%05d',FontScaling='Do not scale fonts')
    except:
        pass  

if paravParam.actual_movement:
    try:
        set_colormap_actualMovement(transform1Display)
        # HideScalarBarIfNotNeeded(brokenBondsLUT, renderView1)
        # save animation
        pathAndName = thisDirectory+'/a_actualMovement.png'
        SaveAnimation(pathAndName, renderView1, ImageResolution=[2054, 1248],
            FrameWindow=rangeForAnimation, 
            # PNG options
            SuffixFormat='_%05d',FontScaling='Do not scale fonts')
    except:
        pass

if paravParam.healing:
    try:
        set_colormap_healing(transform1Display)
        # HideScalarBarIfNotNeeded(brokenBondsLUT, renderView1)
        # save animation
        pathAndName = thisDirectory+'/a_healing.png'
        SaveAnimation(pathAndName, renderView1, ImageResolution=[2054, 1248],
            FrameWindow=rangeForAnimation, 
            # PNG options
            SuffixFormat='_%05d',FontScaling='Do not scale fonts')
    except:
        pass

# toggle 3D widget visibility (only when running from the GUI)
Hide3DWidgets(proxy=transform1.Transform)



# update the view to ensure updated data information
renderView1.Update()
