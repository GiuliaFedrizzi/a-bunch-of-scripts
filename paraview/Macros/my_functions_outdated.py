#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

def applyTableToPoints():
    # Properties modified on tableToPoints1
    tableToPoints1 = TableToPoints()
    tableToPoints1.XColumn = 'x coord'
    tableToPoints1.YColumn = 'y coord'
    tableToPoints1.ZColumn = 'z coord'
    tableToPoints1.a2DPoints = 1
    return tableToPoints1

def edit_transform_properties(transform1Display):
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

def applyLightingAnd2Dview():
    renderView1 = GetActiveViewOrCreate('RenderView')
    LoadPalette(paletteName='WhiteBackground')
    # apply lighting, set the view mode to 2D and camera position
    renderView1.KeyLightElevation = 10
    renderView1.FillLightWarmth = 80
    renderView1.KeyLightIntensity = 0.9
    renderView1.FillLightElevation = -35
    #changing interaction mode based on data extents
    renderView1.InteractionMode = '2D'
    renderView1.CameraPosition = [0.499375, 0.498276665, 10000.0]
    renderView1.CameraFocalPoint = [0.499375, 0.498276665, 0.0]
    renderView1.CameraParallelScale = 0.7046833272972777

def set_colormap_pressure(transform1Display):
    """
    set the fluid pressure and Broken bonds colormaps
    set the representation to Broken Bonds
    """
    # get active view
    renderView1 = GetActiveViewOrCreate('RenderView')
    
    # ========================= PRESSURE ================================================
    # get color transfer function/color map for 'BrokenBonds'
    # set scalar coloring
    ColorBy(transform1Display, ('POINTS', 'Pressure'))

    # rescale color and/or opacity maps used to include current data range
    transform1Display.RescaleTransferFunctionToDataRange(True, False)

    # show color bar/color legend
    transform1Display.SetScalarBarVisibility(renderView1, True)

    # get color transfer function/color map for 'Pressure'
    pressureLUT = GetColorTransferFunction('Pressure')

    # colormap:
    pressureLUT.ApplyPreset('erdc_cyan2orange', True)

    # get opacity transfer function/opacity map for 'Pressure'
    pressurePWF = GetOpacityTransferFunction('Pressure')

    # get color legend/bar for pressureLUT in view renderView1
    pressureLUTColorBar = GetScalarBar(pressureLUT, renderView1)

    pressureLUTColorBar.TitleBold = 1
    #format:
    pressureLUTColorBar.LabelFormat = '%-#6.3e'
    pressureLUTColorBar.RangeLabelFormat = '%-#6.3e'

    # change scalar bar placement
    pressureLUTColorBar.WindowLocation = 'AnyLocation'
    pressureLUTColorBar.Position = [0.7100785340314135, 0.34293193717277487]

    


def set_colormap_broken_bonds(transform1Display):
    renderView1 = GetActiveViewOrCreate('RenderView')
    # ========================= BROKEN BONDS ================================================
    # get color transfer function/color map for 'BrokenBonds'
    brokenBondsLUT = GetColorTransferFunction('BrokenBonds')

    # Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
    brokenBondsLUT.ApplyPreset('pink_Matlab', True)

    # show color bar/color legend
    transform1Display.SetScalarBarVisibility(renderView1, True)

    # set scalar coloring
    ColorBy(transform1Display, ('POINTS', 'Broken Bonds'))

    
    # rescale color and/or opacity maps used to include current data range
    transform1Display.RescaleTransferFunctionToDataRange(True, False)

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

    # hide pressure LUT
    pressureLUT = GetColorTransferFunction('Pressure')
    # Hide the scalar bar for this color map if no visible data is colored by it.
    HideScalarBarIfNotNeeded(pressureLUT, renderView1)


def set_colormap_pressure(transform1Display):
    renderView1 = GetActiveViewOrCreate('RenderView')
    # ========================= BROKEN BONDS ================================================
    # get color transfer function/color map for 'Porosity'
    PorosityLUT = GetColorTransferFunction('Porosity')

    # Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
    PorosityLUT.ApplyPreset('BuPu', True)

    # show color bar/color legend
    transform1Display.SetScalarBarVisibility(renderView1, True)

    # set scalar coloring
    ColorBy(transform1Display, ('POINTS', 'Porosity'))

    
    # rescale color and/or opacity maps used to include current data range
    transform1Display.RescaleTransferFunctionToDataRange(True, False)

    # get color legend/bar for brokenBondsLUT in view renderView1
    porosityLUTColorBar = GetScalarBar(porosityLUT, renderView1)
    porosityLUTColorBar.TitleBold = 1
    porosityLUTColorBar.Title = 'Porosity'
    porosityLUTColorBar.ComponentTitle = '\nMelt Fraction'
    porosityLUTColorBar.TitleFontSize = 21
    porosityLUTColorBar.LabelBold = 1
    porosityLUTColorBar.LabelFontSize = 20

    # Chenge the number format in the broken bonds label
    # brokenBondsLUTColorBar.LabelFormat = '%#.1f'
    # brokenBondsLUTColorBar.RangeLabelFormat = '%#.1f'
    # # and define custom labels (from 0 to 6, which is the max number of bb)
    # brokenBondsLUTColorBar.UseCustomLabels = 1
    # brokenBondsLUTColorBar.CustomLabels = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    # brokenBondsLUTColorBar.ScalarBarThickness = 20

    porosityLUTColorBar.AutomaticLabelFormat = 0
    porosityLUTColorBar.RangeLabelFormat = '%-#6.1g' 


    # Rescale transfer function - set limits
    porosityLUT.RescaleTransferFunction(0.02, 0.3)

    # change scalar bar placement
    porosityLUT.WindowLocation = 'AnyLocation'
    porosityLUT.Position = [0.7700414230019493, 0.35016025641025644]
    porosityLUT.ScalarBarLength = 0.33

    # hide pressure LUT
    pressureLUT = GetColorTransferFunction('Pressure')
    # Hide the scalar bar for this color map if no visible data is colored by it.
    HideScalarBarIfNotNeeded(pressureLUT, renderView1)

    # hide bb LUT
    brokenBondsLUT = GetColorTransferFunction('BrokenBonds')
    # Hide the scalar bar for this color map if no visible data is colored by it.
    HideScalarBarIfNotNeeded(brokenBondsLUT, renderView1)

