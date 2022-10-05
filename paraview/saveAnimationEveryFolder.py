# trace generated using paraview version 5.9.0

#### import the simple module from the paraview
from paraview.simple import *
import glob
import os
import time
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

#LoadPalette(paletteName='WhiteBackground')

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

# get active view
#spreadSheetView1 = GetActiveViewOrCreate('SpreadSheetView')

# get animation scene
animationScene1 = GetAnimationScene()

# update animation scene based on data timesteps
animationScene1.UpdateAnimationUsingDataTimeSteps()

# destroy spreadSheetView1
#Delete(spreadSheetView1)
#del spreadSheetView1

# get layout
layout1 = GetLayoutByName("Layout #1")

# close an empty frame
#layout1.Collapse(2)

# find view
renderView1 = FindViewOrCreate('RenderView1', viewtype='RenderView')

# set active view
SetActiveView(renderView1)

# rename source object
RenameSource('mf0001_freq1e1', my_experiment)


# =========== END OPEN FILES =================

# find source
#my_experiment0 = FindSource('my_experiment0*')

# create a new 'Table To Points'
tableToPoints1 = TableToPoints(registrationName='TableToPoints1')#, Input=my_experiment0)
tableToPoints1.XColumn = 'x coord'
tableToPoints1.YColumn = 'y coord'
tableToPoints1.ZColumn = 'z coord'

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')


#  ====== GLYPH =======
# create a new 'Glyph'
glyph1 = Glyph(registrationName='Glyph1', Input=tableToPoints1,
    GlyphType='Arrow')
glyph1.OrientationArray = ['POINTS', 'No orientation array']
glyph1.ScaleArray = ['POINTS', 'Broken Bonds']
glyph1.ScaleFactor = 0.02
glyph1.GlyphTransform = 'Transform2'

# show data in view
glyph1Display = Show(glyph1, renderView1)

# trace defaults for the display properties.
glyph1Display.Representation = 'Surface'
glyph1Display.ColorArrayName = [None, '']
#glyph1Display.SelectTCoordArray = 'None'
#glyph1Display.SelectNormalArray = 'None'
#glyph1Display.SelectTangentArray = 'None'
glyph1Display.OSPRayScaleArray = 'Broken Bonds'
glyph1Display.OSPRayScaleFunction = 'PiecewiseFunction'
glyph1Display.SelectOrientationVectors = 'None'
glyph1Display.ScaleFactor = 0.09975000023841858
#glyph1Display.SelectScaleArray = 'Broken Bonds'
#glyph1Display.GlyphType = 'Arrow'
glyph1.GlyphType = 'Sphere'
glyph1Display.GlyphTableIndexArray = 'Broken Bonds'
glyph1Display.GaussianRadius = 0.004987500011920929
glyph1Display.SetScaleArray = ['POINTS', 'Broken Bonds']
glyph1.ScaleArray = ['POINTS', 'No scale array']
glyph1Display.ScaleTransferFunction = 'PiecewiseFunction'
#glyph1Display.OpacityArray = ['POINTS', 'Broken Bonds']
glyph1Display.OpacityTransferFunction = 'PiecewiseFunction'
glyph1Display.DataAxesGrid = 'GridAxesRepresentation'
glyph1Display.PolarAxes = 'PolarAxesRepresentation'


glyph1.GlyphMode = 'All Points'

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
glyph1Display.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
glyph1Display.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

# hide data in view
#Hide(tableToPoints1, renderView1)


# Properties modified on glyph1
#glyph1.ScaleFactor = 0.02

# set scalar coloring
ColorBy(glyph1Display, ('POINTS', 'Pressure'))

# rescale color and/or opacity maps used to include current data range
glyph1Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
#glyph1Display.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'Pressure'
pressureLUT = GetColorTransferFunction('Pressure')
pressureLUT.RGBPoints = [1000000.0, 0.231373, 0.298039, 0.752941, 1000064.0, 0.865003, 0.865003, 0.865003, 1000128.0, 0.705882, 0.0156863, 0.14902]
pressureLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'Pressure'
pressurePWF = GetOpacityTransferFunction('Pressure')
pressurePWF.Points = [1000000.0, 0.0, 0.5, 0.0, 1000128.0, 1.0, 0.5, 0.0]
pressurePWF.ScalarRangeInitialized = 1

# reset view to fit data
renderView1.ResetCamera()

# ======= Lighting =========

renderView1.KeyLightElevation = 10
renderView1.FillLightWarmth = 80
renderView1.KeyLightIntensity = 0.9
renderView1.FillLightElevation = -35

# get color legend/bar for pressureLUT in view renderView1
pressureLUTColorBar = GetScalarBar(pressureLUT, renderView1)
pressureLUTColorBar.Title = 'Pressure'
pressureLUTColorBar.ComponentTitle = ''

# change scalar bar placement
pressureLUTColorBar.WindowLocation = 'AnyLocation'
pressureLUTColorBar.Position = [0.7700414230019493, 0.35016025641025644]
pressureLUTColorBar.ScalarBarLength = 0.33

#================================================================
# addendum: following script captures some of the application
# state to faithfully reproduce the visualization during playback
#================================================================

# get layout
layout1 = GetLayout()

#--------------------------------
# saving layout sizes for layouts

# layout/tab size in pixels
#layout1.SetSize(2052, 1248)

#-----------------------------------
# saving camera placements for views

# current camera placement for renderView1
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [0.4987499783746898, 0.4939709787722677, 2.75990303673694]
renderView1.CameraFocalPoint = [0.4987499783746898, 0.4939709787722677, 0.0]
renderView1.CameraParallelScale = 0.714315468543802

# reset view to fit data
renderView1.ResetCamera()
#--------------------------------------------
# uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).

#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# hide data in view
#Hide(glyph1, renderView1)

"""
# ========= CLIP ============= 
# create a new 'Clip'
clip1 = Clip(registrationName='Clip1', Input=glyph1)
#clip1.ClipType = 'Plane'
clip1.ClipType = 'Box'
#clip1.HyperTreeGridClipper = 'Plane'
# Properties modified on clip1.ClipType
clip1.ClipType.Position = [-0.0015, 0.0, -0.01]
clip1.ClipType.Length = [1.0002, 0.989, 0.02]
clip1.Scalars = ['POINTS', 'Break Strength']
clip1.Value = 1.73205081 # ?
"""
# get active view
renderView1 = GetActiveViewOrCreate('RenderView')

# Properties modified on renderView1.AxesGrid
renderView1.AxesGrid.Visibility = 1

# get the material library
materialLibrary1 = GetMaterialLibrary()

# toggle 3D widget visibility (only when running from the GUI)
#Hide3DWidgets(proxy=clip1.ClipType)

# show data in view
glyph1Display = Show(glyph1, renderView1)#, 'UnstructuredGridRepresentation')

# === annotate time filter ===
annotateTimeFilter1 = AnnotateTimeFilter(registrationName='AnnotateTimeFilter1', Input=glyph1)
annotateTimeFilter1Display = Show(annotateTimeFilter1, renderView1)#, 'TextSourceRepresentation')
annotateTimeFilter1.Format = 'Timestep: %9.2e'
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
# get color transfer function/color map for 'BrokenBonds'
brokenBondsLUT = GetColorTransferFunction('BrokenBonds')

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
brokenBondsLUT.ApplyPreset('erdc_pbj_lin', True)

# get opacity transfer function/opacity map for 'BrokenBonds'
brokenBondsPWF = GetOpacityTransferFunction('BrokenBonds')

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
brokenBondsLUT.ApplyPreset('erdc_pbj_lin', True)

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')

# get color legend/bar for brokenBondsLUT in view renderView1
brokenBondsLUTColorBar = GetScalarBar(brokenBondsLUT, renderView1)

# Properties modified on brokenBondsLUTColorBar
brokenBondsLUTColorBar.TitleBold = 1
brokenBondsLUTColorBar.TitleFontSize = 21
brokenBondsLUTColorBar.LabelBold = 1
brokenBondsLUTColorBar.LabelFontSize = 20

# Properties modified on brokenBondsLUTColorBar
brokenBondsLUTColorBar.Title = 'Number of Broken Bonds'
brokenBondsLUTColorBar.ComponentTitle = ''

# Properties modified on brokenBondsLUTColorBar
brokenBondsLUTColorBar.LabelFormat = '%#.1f'
brokenBondsLUTColorBar.RangeLabelFormat = '%#.1f'

# change scalar bar placement
brokenBondsLUTColorBar.WindowLocation = 'AnyLocation'
brokenBondsLUTColorBar.Position = [0.7700414230019493, 0.35016025641025644]
brokenBondsLUTColorBar.ScalarBarLength = 0.33

# Properties modified on brokenBondsLUTColorBar
brokenBondsLUTColorBar.UseCustomLabels = 1
brokenBondsLUTColorBar.CustomLabels = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
brokenBondsLUTColorBar.ScalarBarThickness = 20

ColorBy(glyph1Display,('POINTS','Broken Bonds'))

# no range labels -> I'm using custom ones
brokenBondsLUTColorBar.AddRangeLabels = 0

# Rescale transfer function
brokenBondsLUT.RescaleTransferFunction(0.0, 6.0) # min = 0 bb, max = 6 bb

# get opacity transfer function/opacity map for 'BrokenBonds'
brokenBondsPWF = GetOpacityTransferFunction('BrokenBonds')

# Rescale transfer function
brokenBondsPWF.RescaleTransferFunction(0.0, 6.0)

try:
    HideScalarBarIfNotNeeded(porosityLUT, renderView1)
except Exception:
    pass

try:
    HideScalarBarIfNotNeeded(pressureLUT, renderView1)
except Exception:
    pass
# -----------

# save animation
pathAndName = thisDirectory+'/a_brokenBonds_.png'
SaveAnimation(pathAndName, renderView1, ImageResolution=[2054, 1248],
    FrameWindow=rangeForAnimation, 
    # PNG options
    SuffixFormat='_%05d')

# hide color bar/color legend
#clip1Display.SetScalarBarVisibility(renderView1, False) # not working

# ------ end broken bonds viz map -------


# ------ start pressure viz map -------
# get color transfer function/color map for 'Pressure'
pressureLUT = GetColorTransferFunction('Pressure')
# get color legend/bar for pressureLUT in view renderView1
pressureLUTColorBar = GetScalarBar(pressureLUT, renderView1)

ColorBy(glyph1Display,('POINTS','Pressure'))
HideScalarBarIfNotNeeded(brokenBondsLUT, renderView1)

# Properties modified on brokenBondsLUTColorBar
pressureLUTColorBar.TitleBold = 1
pressureLUTColorBar.TitleFontSize = 21
pressureLUTColorBar.LabelBold = 1
pressureLUTColorBar.LabelFontSize = 20

# Properties modified on brokenBondsLUTColorBar
pressureLUTColorBar.Title = 'Fluid Pressure'
pressureLUTColorBar.ComponentTitle = ''

# toggle 3D widget visibility (only when running from the GUI)
#Hide3DWidgets(proxy=glyph1.ClipType)

# reset view to fit data
renderView1.ResetCamera()

# hide data in view
Hide(glyph1, renderView1)

# show color bar/color legend
glyph1Display.SetScalarBarVisibility(renderView1, True)
# ------ end pressure viz map -------

# axes properties
# Properties modified on renderView1.AxesGrid
renderView1.AxesGrid.XLabelBold = 1
renderView1.AxesGrid.XLabelFontSize = 25
renderView1.AxesGrid.YLabelBold = 1
renderView1.AxesGrid.YLabelFontSize = 25

# Properties modified on renderView1.AxesGrid
renderView1.AxesGrid.XLabelFontSize = 15
renderView1.AxesGrid.YLabelFontSize = 15

# Properties modified on renderView1.AxesGrid
renderView1.AxesGrid.XLabelOpacity = 0.6000000000000001
renderView1.AxesGrid.YLabelOpacity = 0.6000000000000001

# reset view to fit data
renderView1.ResetCamera()

# update the view to ensure updated data information
renderView1.Update()

try:
    HideScalarBarIfNotNeeded(porosityLUT, renderView1)
except Exception:
    pass

try:
    HideScalarBarIfNotNeeded(brokenBondsLUT, renderView1)
except Exception:
    pass

# -----------

# save animation
pathAndName = thisDirectory+'/animation_fluidPressure.png'
SaveAnimation(pathAndName, renderView1, ImageResolution=[2054, 1248],
    FrameWindow=rangeForAnimation, 
    # PNG options
    SuffixFormat='_%05d')
# SaveScreenshot('/Users/giuliafedrizzi/OneDrive - University of Leeds/PhD/meetings/ppt/2021-12-17_xmas/exampleFrac3py.png', renderView1, ImageResolution=[2054, 1248])

