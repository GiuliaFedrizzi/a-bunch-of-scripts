"""
open all csv files in directory
find how often they were saved
find timestep
apply table to points filter
apply delaunay filter (fills the gap between points)
apply transform
colour points by:
- fluid pressure
- number of broken bonds
save animation (=screenshot of every timestep)


Giulia Fedrizzi October 2022
"""

import sys
# add the path to directory with macros to the sys path
sys.path.append('/home/home01/scgf/.config/ParaView/Macros')  # change if path to A_TTPsphBclip.py is different

from my_functions import * # includes paraview simple

import glob
import os
import time

LoadPalette(paletteName='WhiteBackground')

# some variables for saving animations
thisDirectory = os.getcwd()

# ======== OPEN FILES ================
# find files
allFiles = []
for i in sorted(glob.glob("my_experiment*.csv")):
    allFiles.append(os.getcwd() + "/" +i)
    
numberOfFiles = len(allFiles)

rangeForAnimation = [0,numberOfFiles-1] # from 0 to n-1, because 0 is included (and the last number too)


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
my_experiment = CSVReader(FileName=allFiles)


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


# =========== END OPEN FILES =================

applyTableToPoints()

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

# show data in view
transform1Display = Show(transform1, renderView1)

# hide data in view
Hide(delaunay2D1, renderView1)

# === annotate time filter ===
annotateTimeFilter1 = AnnotateTimeFilter(registrationName='AnnotateTimeFilter1')#, Input=glyph1)
annotateTimeFilter1Display = Show(annotateTimeFilter1, renderView1)#, 'TextSourceRepresentation')
annotateTimeFilter1.Format = 'Time: %9.2e'
annotateTimeFilter1.Shift = 0   # first timestep
annotateTimeFilter1.Scale = (time_step_num*frequency)   # time between timesteps, extracted from input.txt, times the frequency for saving files
# location of filter
annotateTimeFilter1Display.WindowLocation = 'AnyLocation'

renderView1.Update()
# ------- POROSITY viz --------
# set properties for porosity
# porosityLUT = GetColorTransferFunction('Porosity')
# porosityLUTColorBar = GetScalarBar(porosityLUT, renderView1)

# # Properties modified on porosityLUTColorBar
# porosityLUTColorBar.TitleBold = 1
# porosityLUTColorBar.TitleFontSize = 21
# porosityLUTColorBar.LabelBold = 1
# porosityLUTColorBar.LabelFontSize = 20

# # Properties modified on porosityLUTColorBar
# porosityLUTColorBar.AutomaticLabelFormat = 0
# porosityLUTColorBar.RangeLabelFormat = '%-#6.1g'

# # Properties modified on porosityLUTColorBar
# porosityLUTColorBar.AutomaticLabelFormat = 1

# # Properties modified on porosityLUTColorBar
# porosityLUTColorBar.Title = 'Porosity'
# porosityLUTColorBar.ComponentTitle = '\nMelt Fraction'

# # change scalar bar placement
# porosityLUTColorBar.WindowLocation = 'AnyLocation'
# porosityLUTColorBar.Position = [0.7700414230019493, 0.35016025641025644]
# porosityLUTColorBar.ScalarBarLength = 0.33

# ColorBy(glyph1Display,('POINTS','Porosity'))

# # Rescale - set limits
# porosityLUT.RescaleTransferFunction(0.02, 0.3)

# # get opacity transfer function/opacity map for 'Porosity'
# porosityPWF = GetOpacityTransferFunction('Porosity')

# # Rescale transfer function
# porosityPWF.RescaleTransferFunction(0.02, 0.3)

# # -----------

# # save animation
# pathAndName = thisDirectory+'/animation_porosity.png'
# SaveAnimation(pathAndName, renderView1, ImageResolution=[2054, 1248],
#     FrameWindow=rangeForAnimation, 
#     # PNG options
#     SuffixFormat='_%05d')

# ------ start broken bonds viz map -------

brokenBondsLUT = set_colormap_broken_bonds(transform1Display)

# hide possible colorbars
try:
    HideScalarBarIfNotNeeded(porosityLUT, renderView1)
except Exception:
    pass

try:
    HideScalarBarIfNotNeeded(pressureLUT, renderView1)
except Exception:
    pass

# save animation
pathAndName = thisDirectory+'/a_brokenBonds.png'
SaveAnimation(pathAndName, renderView1, ImageResolution=[2054, 1248],
    FrameWindow=rangeForAnimation, 
    # PNG options
    SuffixFormat='_%05d')

# ------ end broken bonds viz map -------
# 
pressureLUT = set_colormap_pressure(transform1Display)

# hide possible colorbars
try:
    HideScalarBarIfNotNeeded(porosityLUT, renderView1)
except Exception:
    pass

try:
    HideScalarBarIfNotNeeded(brokenBondsLUT, renderView1)
except Exception:
    pass

# update the view to ensure updated data information
renderView1.Update()

# save animation
pathAndName = thisDirectory+'/a_fluidPressure.png'
SaveAnimation(pathAndName, renderView1, ImageResolution=[2054, 1248],
    FrameWindow=rangeForAnimation, 
    # PNG options
    SuffixFormat='_%05d')

