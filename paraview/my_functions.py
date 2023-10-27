#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

def applyTableToPoints(i):
    # Properties modified on tableToPain
    #k.
    tableToPain = TableToPoints(Input=i)
    #tableToPoints13 = TableToPoints(Input=nobackupscgfmyExperimentsgaussScaleFixFrac2noSize200size050)
    tableToPain.XColumn = 'x coord'
    tableToPain.YColumn = 'y coord'
    tableToPain.ZColumn = 'z coord'
    tableToPain.a2DPoints = 1
def applyTableToPointsSingle():
    tableToPain = TableToPoints()
    tableToPain.XColumn = 'x coord'
    tableToPain.YColumn = 'y coord'
    tableToPain.ZColumn = 'z coord'
    tableToPain.a2DPoints = 1

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
    
    # hide potential colormaps that are not being used
    hide_colormaps()

    # get color legend/bar for pressureLUT in view renderView1
    pressureLUTColorBar = GetScalarBar(pressureLUT, renderView1)

    pressureLUTColorBar.TitleBold = 1
    #format:
    pressureLUTColorBar.LabelFormat = '%-#6.3e'
    pressureLUTColorBar.RangeLabelFormat = '%-#6.3e'

    # change scalar bar placement
    pressureLUTColorBar.WindowLocation = 'AnyLocation'
    pressureLUTColorBar.Position = [0.7100785340314135, 0.34293193717277487]

def hide_colormaps():
    """Hide the scalar bar for this color map if no visible data is colored by it."""
    renderView1 = GetActiveViewOrCreate('RenderView')
    try:
        # hide pressure LUT
        pressureLUT = GetColorTransferFunction('Pressure')
        HideScalarBarIfNotNeeded(pressureLUT, renderView1)
    except:
        pass

    try:
        # hide bb LUT
        brokenBondsLUT = GetColorTransferFunction('BrokenBonds')
        HideScalarBarIfNotNeeded(brokenBondsLUT, renderView1)
    except:
        pass

    try:
        # hide poro LUT
        porosityLUT = GetColorTransferFunction('Porosity')
        HideScalarBarIfNotNeeded(porosityLUT, renderView1)
    except:
        pass

    try:
        # hide mean stress LUT
        meanStressLUT = GetColorTransferFunction('MeanStress')
        HideScalarBarIfNotNeeded(meanStressLUT, renderView1)
    except:
        pass

    try:
        # hide actual Movement LUT
        actualMovementLUT = GetColorTransferFunction('Actual Movement')
        HideScalarBarIfNotNeeded(actualMovementLUT, renderView1)
    except:
        pass

    try:
        # hide healing LUT
        healingLUT = GetColorTransferFunction('Healing')
        HideScalarBarIfNotNeeded(healingLUT, renderView1)
    except:
        pass

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
    brokenBondsLUTColorBar.TitleFontSize = 21
    brokenBondsLUTColorBar.LabelFontSize = 20

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

    hide_colormaps()
    # # hide pressure LUT
    # pressureLUT = GetColorTransferFunction('Pressure')
    # # Hide the scalar bar for this color map if no visible data is colored by it.
    # HideScalarBarIfNotNeeded(pressureLUT, renderView1)


def set_colormap_porosity(transform1Display):
    renderView1 = GetActiveViewOrCreate('RenderView')
    # ========================= BROKEN BONDS ================================================
    # get color transfer function/color map for 'Porosity'
    porosityLUT = GetColorTransferFunction('Porosity')

    # Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
    # porosityLUT.ApplyPreset('BuPu', True)   # bad colours
    #porosityLUT.ApplyPreset('Gray and Red', True)

    #RGBPoints is a list of doubles in the order (value_0, red_0, green_0, blue_0, value_1, red_1, green_1, blue_1, ...)
    # porosityLUT.RGBPoints = [0.0383252, 0.101961, 0.101961, 0.101961, 0.060131396515999996, 0.227451, 0.227451, 0.227451, 0.08193759303199999, 0.359939, 0.359939, 0.359939, 0.10374396331639998, 0.502653, 0.502653, 0.502653, 0.12555015983239998, 0.631373, 0.631373, 0.631373, 0.1473563563484, 0.749865, 0.749865, 0.749865, 0.1691625528644, 0.843368, 0.843368, 0.843368, 0.19096874938039998, 0.926105, 0.926105, 0.926105, 0.22411532700061798, 0.999846, 0.997232, 0.995694, 0.2448800951242447, 0.994925, 0.908651, 0.857901, 0.2689235210418701, 0.982468, 0.800692, 0.706113, 0.2875025272369385, 0.960323, 0.66782, 0.536332, 0.3104530870914459, 0.894579, 0.503806, 0.399769, 0.3323107361793518, 0.81707, 0.33218, 0.281046, 0.3530755341053009, 0.728489, 0.155017, 0.197386, 0.36541866904520004, 0.576932, 0.055363, 0.14925, 0.385862, 0.403922, 0.0, 0.121569]
    # show color bar/color legend
    transform1Display.SetScalarBarVisibility(renderView1, True)

    # set scalar coloring
    ColorBy(transform1Display, ('POINTS', 'Porosity'))

    
    # rescale color and/or opacity maps used to include current data range
    # transform1Display.RescaleTransferFunctionToDataRange(True, False)

    # get range of the Porosity array
    # s = GetActiveSource()
    # poro_range = s.PointData.GetArray("Porosity").GetRange()   # will only give me the range of the first one...

    # bounds for the porosity range
    # lower_bound = poro_range[0] 
    # upper_bound = poro_range[1] 

    lower_bound = 0.28
    upper_bound = 0.39

    intermediate = lower_bound + 0.8*(upper_bound-lower_bound) #  the third RGB point, 0.72 between lower and upper
    #  p03, t_ref = 80
    # porosityLUT.RGBPoints = [0.23, 0.101961, 0.101961, 0.101961, 0.2552606165409088, 0.843368, 0.843368, 0.843368, 0.277074, 0.403922, 0.0, 0.121569]
    
    #  p03, t_ref = 140
    porosityLUT.RGBPoints = [lower_bound, 0.101961, 0.101961, 0.101961, intermediate, 0.843368, 0.843368, 0.843368, upper_bound, 0.403922, 0.0, 0.121569]
    # get color legend/bar for brokenBondsLUT in view renderView1
    porosityLUTColorBar = GetScalarBar(porosityLUT, renderView1)
    porosityLUTColorBar.TitleBold = 1
    porosityLUTColorBar.Title = 'Porosity'
    porosityLUTColorBar.ComponentTitle = '\n(Melt Fraction)'
    porosityLUTColorBar.TitleFontSize = 21
    porosityLUTColorBar.LabelBold = 1
    porosityLUTColorBar.LabelFontSize = 20


    # porosityLUTColorBar.AutomaticLabelFormat = 0
    porosityLUTColorBar.RangeLabelFormat = '%-#6.4g' 
    

    # Rescale transfer function - set limits
    # porosityLUT.RescaleTransferFunction(0.02, 0.3)

    # change scalar bar placement
    porosityLUTColorBar.WindowLocation = 'AnyLocation'
    porosityLUTColorBar.Position = [0.7700414230019493, 0.35016025641025644]
    porosityLUTColorBar.ScalarBarLength = 0.33

    hide_colormaps()

def set_colormap_mean_stress(transform1Display):
    """
    set the mean stress colormap
    """
    # get active view
    renderView1 = GetActiveViewOrCreate('RenderView')
    # get color transfer function/color map for 'MeanStress'
    # set scalar coloring
    ColorBy(transform1Display, ('POINTS', 'MeanStress'))

    # rescale color and/or opacity maps used to include current data range
    transform1Display.RescaleTransferFunctionToDataRange(True, False)

    # show color bar/color legend
    transform1Display.SetScalarBarVisibility(renderView1, True)

    # get color transfer function/color map for 'MeanStress'
    meanStressLUT = GetColorTransferFunction('MeanStress')

    # colormap:
    #meanStressLUT.ApplyPreset('erdc_cyan2orange', True)

    # get opacity transfer function/opacity map for 'Pressure'
    meanStressPWF = GetOpacityTransferFunction('meanStress')

    # get color legend/bar for meanStressLUT in view renderView1
    meanStressLUTColorBar = GetScalarBar(meanStressLUT, renderView1)

    meanStressLUTColorBar.TitleBold = 1
    #format:
    meanStressLUTColorBar.LabelFormat = '%-#6.3e'
    meanStressLUTColorBar.RangeLabelFormat = '%-#6.3e'

    # change scalar bar placement
    meanStressLUTColorBar.WindowLocation = 'AnyLocation'
    meanStressLUTColorBar.Position = [0.7100785340314135, 0.34293193717277487]

    hide_colormaps()

def set_colormap_actualMovement(transform1Display):
    '''
    set the actual movement colormap
    '''
    renderView1 = GetActiveViewOrCreate('RenderView')

    # set scalar coloring
    ColorBy(transform1Display, ('POINTS', 'Actual Movement'))

    # rescale color and/or opacity maps used to include current data range
    transform1Display.RescaleTransferFunctionToDataRange(True, False)

    # show color bar/color legend
    transform1Display.SetScalarBarVisibility(renderView1, True)

    # get color transfer function/color map for 'ActualMovement'
    actualMovementLUT = GetColorTransferFunction('ActualMovement')

    # get opacity transfer function/opacity map for 'ActualMovement'
    actualMovementPWF = GetOpacityTransferFunction('ActualMovement')

    hide_colormaps()

def set_colormap_healing(transform1Display,max_time):
    '''
    set the healing colormap
    '''
    renderView1 = GetActiveViewOrCreate('RenderView')
    # set scalar coloring
    ColorBy(transform1Display, ('POINTS', 'Healing'))

    # rescale color and/or opacity maps used to include current data range
    transform1Display.RescaleTransferFunctionToDataRange(True, False)

    # show color bar/color legend
    transform1Display.SetScalarBarVisibility(renderView1, True)
    # -----------------

    # get color transfer function/color map for 'Healing'
    healingLUT = GetColorTransferFunction('Healing')

    # get opacity transfer function/opacity map for 'Healing'
    healingPWF = GetOpacityTransferFunction('Healing')

    # Apply a preset using its name.
    healingLUT.ApplyPreset('CIELab Blue to Red', True)

    #  change range for colormap
    healingLUT.RescaleTransferFunction(0.0, max_time) # min = 0, max = max time
    print("max_time is "+str(max_time))

    # hide other colmaps
    hide_colormaps()
    

def get_scale():
    try:
        domain_size = float(GetActiveSource().FileName[0].split("/")[-2].replace('size',''))
        return domain_size, 1 #  domain size = 1, found_scale = 1 (it found a scale)
    except:
        print("No scale found")
        return 1.0, 0   #  domain size = 1, found_scale = 0 (it didn't find a scale)
    else:
        print("Scale: ",domain_size)

