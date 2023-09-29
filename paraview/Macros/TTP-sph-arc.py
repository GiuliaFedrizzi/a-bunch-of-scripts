# trace generated using paraview version 5.9.0

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

LoadPalette(paletteName='WhiteBackground')

# create a new 'Table To Points'
tableToPoints1 = TableToPoints(registrationName='TableToPoints1')#, Input=my_experiment30000csv)
tableToPoints1.XColumn = 'x coord'
tableToPoints1.YColumn = 'y coord'
tableToPoints1.ZColumn = 'z coord'
tableToPoints1.a2DPoints = 0
tableToPoints1.KeepAllDataArrays = 0

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')

# create a new 'Glyph'
glyph1 = Glyph(registrationName='Glyph1', Input=tableToPoints1,
    GlyphType='Sphere')
glyph1.OrientationArray = ['POINTS', 'No orientation array']
glyph1.ScaleArray = ['POINTS', 'Broken Bonds']
glyph1.ScaleFactor = 0.09975
glyph1.GlyphTransform = 'Transform2'

# show data in view
glyph1Display = Show(glyph1, renderView1)

# trace defaults for the display properties.
glyph1Display.Representation = 'Surface'
glyph1Display.ColorArrayName = [None, '']
"""glyph1Display.SelectTCoordArray = 'None'
glyph1Display.SelectNormalArray = 'None'
glyph1Display.SelectTangentArray = 'None'
glyph1Display.OSPRayScaleArray = 'Broken Bonds'
glyph1Display.OSPRayScaleFunction = 'PiecewiseFunction'
glyph1Display.SelectOrientationVectors = 'None'
glyph1Display.ScaleFactor = 0.09975000023841858
glyph1Display.SelectScaleArray = 'Broken Bonds'
#glyph1Display.GlyphType = 'Arrow'
"""
glyph1.GlyphType = 'Sphere'
glyph1Display.GlyphTableIndexArray = 'Broken Bonds'
glyph1Display.GaussianRadius = 0.004987500011920929
#glyph1Display.SetScaleArray = ['POINTS', 'Broken Bonds']
glyph1.ScaleArray = ['POINTS', 'No scale array']
glyph1Display.ScaleTransferFunction = 'PiecewiseFunction'
glyph1Display.OpacityArray = ['POINTS', 'Broken Bonds']
glyph1Display.OpacityTransferFunction = 'PiecewiseFunction'
glyph1Display.DataAxesGrid = 'GridAxesRepresentation'
glyph1Display.PolarAxes = 'PolarAxesRepresentation'
glyph1.GlyphMode = 'All Points'

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
glyph1Display.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
glyph1Display.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]


#================================================================
# addendum: following script captures some of the application
# state to faithfully reproduce the visualization during playback
#================================================================

# get layout
layout1 = GetLayout()

#--------------------------------
# saving layout sizes for layouts

# layout/tab size in pixels
#layout1.SetSize(2050, 1248)

#-----------------------------------
# saving camera placements for views

# current camera placement for renderView1
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [0.498754985, 0.49726838175, 2.7152981189802237]
renderView1.CameraFocalPoint = [0.498754985, 0.49726838175, 0.0]
renderView1.CameraParallelScale = 0.5808023688620926

#--------------------------------------------
# uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).


