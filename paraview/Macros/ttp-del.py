# trace generated using paraview version 5.6.0
#
# To ensure correct image size when batch processing, please search 
# for and uncomment the line `# renderView*.ViewSize = [*,*]`


#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()


LoadPalette(paletteName='WhiteBackground')


# Properties modified on tableToPoints1
tableToPoints1 = TableToPoints()
tableToPoints1.XColumn = 'x coord'
tableToPoints1.YColumn = 'y coord'
tableToPoints1.ZColumn = 'z coord'
tableToPoints1.a2DPoints = 1


# get active view
renderView1 = GetActiveViewOrCreate('RenderView')
# ======= Lighting =========


renderView1.KeyLightElevation = 10
renderView1.FillLightWarmth = 80
renderView1.KeyLightIntensity = 0.9
renderView1.FillLightElevation = -35


#changing interaction mode based on data extents
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [0.499375, 0.498276665, 10000.0]
renderView1.CameraFocalPoint = [0.499375, 0.498276665, 0.0]


# update the view to ensure updated data information
renderView1.Update()


# create a new 'Delaunay 2D'
delaunay2D1 = Delaunay2D()


# show data in view
delaunay2D1Display = Show(delaunay2D1, renderView1)


# trace defaults for the display properties.
delaunay2D1Display.Representation = 'Surface'
delaunay2D1Display.ColorArrayName = [None, '']
delaunay2D1Display.OSPRayScaleArray = 'Broken Bonds'
delaunay2D1Display.OSPRayScaleFunction = 'PiecewiseFunction'
delaunay2D1Display.SelectOrientationVectors = 'Broken Bonds'
delaunay2D1Display.ScaleFactor = 0.099875
delaunay2D1Display.SelectScaleArray = 'Broken Bonds'
delaunay2D1Display.GlyphType = 'Arrow'
delaunay2D1Display.GlyphTableIndexArray = 'Broken Bonds'
delaunay2D1Display.GaussianRadius = 0.00499375
delaunay2D1Display.SetScaleArray = ['POINTS', 'Broken Bonds']
delaunay2D1Display.ScaleTransferFunction = 'PiecewiseFunction'
delaunay2D1Display.OpacityArray = ['POINTS', 'Broken Bonds']
delaunay2D1Display.OpacityTransferFunction = 'PiecewiseFunction'
delaunay2D1Display.DataAxesGrid = 'GridAxesRepresentation'
delaunay2D1Display.SelectionCellLabelFontFile = ''
delaunay2D1Display.SelectionPointLabelFontFile = ''
delaunay2D1Display.PolarAxes = 'PolarAxesRepresentation'


# init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
delaunay2D1Display.DataAxesGrid.XTitleFontFile = ''
delaunay2D1Display.DataAxesGrid.YTitleFontFile = ''
delaunay2D1Display.DataAxesGrid.ZTitleFontFile = ''
delaunay2D1Display.DataAxesGrid.XLabelFontFile = ''
delaunay2D1Display.DataAxesGrid.YLabelFontFile = ''
delaunay2D1Display.DataAxesGrid.ZLabelFontFile = ''


# init the 'PolarAxesRepresentation' selected for 'PolarAxes'
delaunay2D1Display.PolarAxes.PolarAxisTitleFontFile = ''
delaunay2D1Display.PolarAxes.PolarAxisLabelFontFile = ''
delaunay2D1Display.PolarAxes.LastRadialAxisTextFontFile = ''
delaunay2D1Display.PolarAxes.SecondaryRadialAxesTextFontFile = ''


# hide data in view
#Hide(tableToPoints1, renderView1)


# update the view to ensure updated data information
renderView1.Update()


# create a new 'Transform'
transform1 = Transform(Input=delaunay2D1)
transform1.Transform = 'Transform'


# show data in view
transform1Display = Show(transform1, renderView1)


# trace defaults for the display properties.
transform1Display.Representation = 'Surface'
transform1Display.ColorArrayName = [None, '']
transform1Display.OSPRayScaleArray = 'Broken Bonds'
transform1Display.OSPRayScaleFunction = 'PiecewiseFunction'
transform1Display.SelectOrientationVectors = 'Broken Bonds'
transform1Display.ScaleFactor = 0.099875
transform1Display.SelectScaleArray = 'Broken Bonds'
transform1Display.GlyphType = 'Arrow'
transform1Display.GlyphTableIndexArray = 'Broken Bonds'
transform1Display.GaussianRadius = 0.00499375
transform1Display.SetScaleArray = ['POINTS', 'Broken Bonds']
transform1Display.ScaleTransferFunction = 'PiecewiseFunction'
transform1Display.OpacityArray = ['POINTS', 'Broken Bonds']
transform1Display.OpacityTransferFunction = 'PiecewiseFunction'
transform1Display.DataAxesGrid = 'GridAxesRepresentation'
transform1Display.SelectionCellLabelFontFile = ''
transform1Display.SelectionPointLabelFontFile = ''
transform1Display.PolarAxes = 'PolarAxesRepresentation'


# init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
transform1Display.DataAxesGrid.XTitleFontFile = ''
transform1Display.DataAxesGrid.YTitleFontFile = ''
transform1Display.DataAxesGrid.ZTitleFontFile = ''
transform1Display.DataAxesGrid.XLabelFontFile = ''
transform1Display.DataAxesGrid.YLabelFontFile = ''
transform1Display.DataAxesGrid.ZLabelFontFile = ''


# init the 'PolarAxesRepresentation' selected for 'PolarAxes'
transform1Display.PolarAxes.PolarAxisTitleFontFile = ''
transform1Display.PolarAxes.PolarAxisLabelFontFile = ''
transform1Display.PolarAxes.LastRadialAxisTextFontFile = ''
transform1Display.PolarAxes.SecondaryRadialAxesTextFontFile = ''


# hide data in view
Hide(delaunay2D1, renderView1)


# update the view to ensure updated data information
renderView1.Update()


# set scalar coloring
ColorBy(transform1Display, ('POINTS', 'Pressure'))


# rescale color and/or opacity maps used to include current data range
transform1Display.RescaleTransferFunctionToDataRange(True, False)


# show color bar/color legend
transform1Display.SetScalarBarVisibility(renderView1, True)


# get color transfer function/color map for 'Pressure'
pressureLUT = GetColorTransferFunction('Pressure')


# get opacity transfer function/opacity map for 'Pressure'
pressurePWF = GetOpacityTransferFunction('Pressure')


# get color legend/bar for pressureLUT in view renderView1
pressureLUTColorBar = GetScalarBar(pressureLUT, renderView1)


# change scalar bar placement
pressureLUTColorBar.WindowLocation = 'AnyLocation'
pressureLUTColorBar.Position = [0.7100785340314135, 0.34293193717277487]


# ========================= BROKEN BONDS ================


# get color transfer function/color map for 'BrokenBonds'
brokenBondsLUT = GetColorTransferFunction('BrokenBonds')


# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
brokenBondsLUT.ApplyPreset('erdc_pbj_lin', True)



# show color bar/color legend
glyph1Display.SetScalarBarVisibility(renderView1, True)


# set scalar coloring
ColorBy(glyph1Display, ('POINTS', 'Broken Bonds'))


# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(porosityLUT, renderView1)


# rescale color and/or opacity maps used to include current data range
glyph1Display.RescaleTransferFunctionToDataRange(True, False)


# get color legend/bar for brokenBondsLUT in view renderView1
brokenBondsLUTColorBar = GetScalarBar(brokenBondsLUT, renderView1)
brokenBondsLUTColorBar.Title = 'Number of Broken Bonds'
brokenBondsLUTColorBar.ComponentTitle = ''
brokenBondsLUTColorBar.TitleBold = 1
brokenBondsLUTColorBar.LabelBold = 1


# Chenge the number format in the broken bonds label
brokenBondsLUTColorBar.LabelFormat = '%#.1f'
brokenBondsLUTColorBar.RangeLabelFormat = '%#.1f'
# and define custom labels (from 0 to 6, which is the max number of bb)
brokenBondsLUTColorBar.UseCustomLabels = 1
brokenBondsLUTColorBar.CustomLabels = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
brokenBondsLUTColorBar.ScalarBarThickness = 20


# no range labels -> I'm using custom ones
brokenBondsLUTColorBar.AddRangeLabels = 0


# Rescale transfer function
brokenBondsLUT.RescaleTransferFunction(0.0, 6.0) # min = 0 bb, max = 6 bb


# change scalar bar placement
brokenBondsLUTColorBar.WindowLocation = 'AnyLocation'
brokenBondsLUTColorBar.Position = [0.7700414230019493, 0.35016025641025644]
brokenBondsLUTColorBar.ScalarBarLength = 0.33


# toggle 3D widget visibility (only when running from the GUI)
Hide3DWidgets(proxy=transform1.Transform)


#### saving camera placements for all active views


# current camera placement for renderView1
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [0.499375, 0.498276665, 10000.0]
renderView1.CameraFocalPoint = [0.499375, 0.498276665, 0.0]
renderView1.CameraParallelScale = 0.7046833272972777


#### uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).


# trace generated using paraview version 5.6.0
#
# To ensure correct image size when batch processing, please search 
# for and uncomment the line `# renderView*.ViewSize = [*,*]`


#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()


LoadPalette(paletteName='WhiteBackground')


# Properties modified on tableToPoints1
tableToPoints1 = TableToPoints()
tableToPoints1.XColumn = 'x coord'
tableToPoints1.YColumn = 'y coord'
tableToPoints1.ZColumn = 'z coord'
tableToPoints1.a2DPoints = 1


# get active view
renderView1 = GetActiveViewOrCreate('RenderView')
# ======= Lighting =========


renderView1.KeyLightElevation = 10
renderView1.FillLightWarmth = 80
renderView1.KeyLightIntensity = 0.9
renderView1.FillLightElevation = -35


#changing interaction mode based on data extents
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [0.499375, 0.498276665, 10000.0]
renderView1.CameraFocalPoint = [0.499375, 0.498276665, 0.0]


# update the view to ensure updated data information
renderView1.Update()


# create a new 'Delaunay 2D'
delaunay2D1 = Delaunay2D()


# show data in view
delaunay2D1Display = Show(delaunay2D1, renderView1)


# trace defaults for the display properties.
delaunay2D1Display.Representation = 'Surface'
delaunay2D1Display.ColorArrayName = [None, '']
delaunay2D1Display.OSPRayScaleArray = 'Broken Bonds'
delaunay2D1Display.OSPRayScaleFunction = 'PiecewiseFunction'
delaunay2D1Display.SelectOrientationVectors = 'Broken Bonds'
delaunay2D1Display.ScaleFactor = 0.099875
delaunay2D1Display.SelectScaleArray = 'Broken Bonds'
delaunay2D1Display.GlyphType = 'Arrow'
delaunay2D1Display.GlyphTableIndexArray = 'Broken Bonds'
delaunay2D1Display.GaussianRadius = 0.00499375
delaunay2D1Display.SetScaleArray = ['POINTS', 'Broken Bonds']
delaunay2D1Display.ScaleTransferFunction = 'PiecewiseFunction'
delaunay2D1Display.OpacityArray = ['POINTS', 'Broken Bonds']
delaunay2D1Display.OpacityTransferFunction = 'PiecewiseFunction'
delaunay2D1Display.DataAxesGrid = 'GridAxesRepresentation'
delaunay2D1Display.SelectionCellLabelFontFile = ''
delaunay2D1Display.SelectionPointLabelFontFile = ''
delaunay2D1Display.PolarAxes = 'PolarAxesRepresentation'


# init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
delaunay2D1Display.DataAxesGrid.XTitleFontFile = ''
delaunay2D1Display.DataAxesGrid.YTitleFontFile = ''
delaunay2D1Display.DataAxesGrid.ZTitleFontFile = ''
delaunay2D1Display.DataAxesGrid.XLabelFontFile = ''
delaunay2D1Display.DataAxesGrid.YLabelFontFile = ''
delaunay2D1Display.DataAxesGrid.ZLabelFontFile = ''


# init the 'PolarAxesRepresentation' selected for 'PolarAxes'
delaunay2D1Display.PolarAxes.PolarAxisTitleFontFile = ''
delaunay2D1Display.PolarAxes.PolarAxisLabelFontFile = ''
delaunay2D1Display.PolarAxes.LastRadialAxisTextFontFile = ''
delaunay2D1Display.PolarAxes.SecondaryRadialAxesTextFontFile = ''


# hide data in view
#Hide(tableToPoints1, renderView1)


# update the view to ensure updated data information
renderView1.Update()


# create a new 'Transform'
transform1 = Transform(Input=delaunay2D1)
transform1.Transform = 'Transform'


# show data in view
transform1Display = Show(transform1, renderView1)


# trace defaults for the display properties.
transform1Display.Representation = 'Surface'
transform1Display.ColorArrayName = [None, '']
transform1Display.OSPRayScaleArray = 'Broken Bonds'
transform1Display.OSPRayScaleFunction = 'PiecewiseFunction'
transform1Display.SelectOrientationVectors = 'Broken Bonds'
transform1Display.ScaleFactor = 0.099875
transform1Display.SelectScaleArray = 'Broken Bonds'
transform1Display.GlyphType = 'Arrow'
transform1Display.GlyphTableIndexArray = 'Broken Bonds'
transform1Display.GaussianRadius = 0.00499375
transform1Display.SetScaleArray = ['POINTS', 'Broken Bonds']
transform1Display.ScaleTransferFunction = 'PiecewiseFunction'
transform1Display.OpacityArray = ['POINTS', 'Broken Bonds']
transform1Display.OpacityTransferFunction = 'PiecewiseFunction'
transform1Display.DataAxesGrid = 'GridAxesRepresentation'
transform1Display.SelectionCellLabelFontFile = ''
transform1Display.SelectionPointLabelFontFile = ''
transform1Display.PolarAxes = 'PolarAxesRepresentation'


# init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
transform1Display.DataAxesGrid.XTitleFontFile = ''
transform1Display.DataAxesGrid.YTitleFontFile = ''
transform1Display.DataAxesGrid.ZTitleFontFile = ''
transform1Display.DataAxesGrid.XLabelFontFile = ''
transform1Display.DataAxesGrid.YLabelFontFile = ''
transform1Display.DataAxesGrid.ZLabelFontFile = ''


# init the 'PolarAxesRepresentation' selected for 'PolarAxes'
transform1Display.PolarAxes.PolarAxisTitleFontFile = ''
transform1Display.PolarAxes.PolarAxisLabelFontFile = ''
transform1Display.PolarAxes.LastRadialAxisTextFontFile = ''
transform1Display.PolarAxes.SecondaryRadialAxesTextFontFile = ''


# hide data in view
Hide(delaunay2D1, renderView1)


# update the view to ensure updated data information
renderView1.Update()


# set scalar coloring
ColorBy(transform1Display, ('POINTS', 'Pressure'))


# rescale color and/or opacity maps used to include current data range
transform1Display.RescaleTransferFunctionToDataRange(True, False)


# show color bar/color legend
transform1Display.SetScalarBarVisibility(renderView1, True)


# get color transfer function/color map for 'Pressure'
pressureLUT = GetColorTransferFunction('Pressure')


# get opacity transfer function/opacity map for 'Pressure'
pressurePWF = GetOpacityTransferFunction('Pressure')


# get color legend/bar for pressureLUT in view renderView1
pressureLUTColorBar = GetScalarBar(pressureLUT, renderView1)


# change scalar bar placement
pressureLUTColorBar.WindowLocation = 'AnyLocation'
pressureLUTColorBar.Position = [0.7100785340314135, 0.34293193717277487]


# ========================= BROKEN BONDS ================


# get color transfer function/color map for 'BrokenBonds'
brokenBondsLUT = GetColorTransferFunction('BrokenBonds')


# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
brokenBondsLUT.ApplyPreset('erdc_pbj_lin', True)



# show color bar/color legend
glyph1Display.SetScalarBarVisibility(renderView1, True)


# set scalar coloring
ColorBy(glyph1Display, ('POINTS', 'Broken Bonds'))


# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(porosityLUT, renderView1)


# rescale color and/or opacity maps used to include current data range
glyph1Display.RescaleTransferFunctionToDataRange(True, False)


# get color legend/bar for brokenBondsLUT in view renderView1
brokenBondsLUTColorBar = GetScalarBar(brokenBondsLUT, renderView1)
brokenBondsLUTColorBar.Title = 'Number of Broken Bonds'
brokenBondsLUTColorBar.ComponentTitle = ''
brokenBondsLUTColorBar.TitleBold = 1
brokenBondsLUTColorBar.LabelBold = 1


# Chenge the number format in the broken bonds label
brokenBondsLUTColorBar.LabelFormat = '%#.1f'
brokenBondsLUTColorBar.RangeLabelFormat = '%#.1f'
# and define custom labels (from 0 to 6, which is the max number of bb)
brokenBondsLUTColorBar.UseCustomLabels = 1
brokenBondsLUTColorBar.CustomLabels = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
brokenBondsLUTColorBar.ScalarBarThickness = 20


# no range labels -> I'm using custom ones
brokenBondsLUTColorBar.AddRangeLabels = 0


# Rescale transfer function
brokenBondsLUT.RescaleTransferFunction(0.0, 6.0) # min = 0 bb, max = 6 bb


# change scalar bar placement
brokenBondsLUTColorBar.WindowLocation = 'AnyLocation'
brokenBondsLUTColorBar.Position = [0.7700414230019493, 0.35016025641025644]
brokenBondsLUTColorBar.ScalarBarLength = 0.33


# toggle 3D widget visibility (only when running from the GUI)
Hide3DWidgets(proxy=transform1.Transform)


#### saving camera placements for all active views


# current camera placement for renderView1
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [0.499375, 0.498276665, 10000.0]
renderView1.CameraFocalPoint = [0.499375, 0.498276665, 0.0]
renderView1.CameraParallelScale = 0.7046833272972777


#### uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).
