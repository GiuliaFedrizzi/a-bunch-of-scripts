#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# find source
#p13_rf1005_dis3 = FindSource('p13_rf1-005_dis3')

# create a new 'Table To Points'
tableToPoints1 = TableToPoints(registrationName='TableToPoints1')
tableToPoints1.XColumn = 'x coord'
tableToPoints1.YColumn = 'y coord'
tableToPoints1.ZColumn = 'z coord'
tableToPoints1.a2DPoints = 0
tableToPoints1.KeepAllDataArrays = 0


# get active view
renderView1 = GetActiveViewOrCreate('RenderView')
# uncomment following to set a specific view size
# renderView1.ViewSize = [1611, 763]
"""
# show data in view
tableToPoints1Display = Show(tableToPoints1, renderView1)
# trace defaults for the display properties.
tableToPoints1Display.Representation = 'Surface'
tableToPoints1Display.AmbientColor = [1.0, 1.0, 1.0]
tableToPoints1Display.ColorArrayName = [None, '']
tableToPoints1Display.DiffuseColor = [1.0, 1.0, 1.0]
tableToPoints1Display.LookupTable = None
tableToPoints1Display.MapScalars = 1
tableToPoints1Display.InterpolateScalarsBeforeMapping = 1
tableToPoints1Display.Opacity = 1.0
tableToPoints1Display.PointSize = 2.0
tableToPoints1Display.LineWidth = 1.0
tableToPoints1Display.Interpolation = 'Gouraud'
tableToPoints1Display.Specular = 0.0
tableToPoints1Display.SpecularColor = [1.0, 1.0, 1.0]
tableToPoints1Display.SpecularPower = 100.0
tableToPoints1Display.Ambient = 0.0
tableToPoints1Display.Diffuse = 1.0
tableToPoints1Display.EdgeColor = [0.0, 0.0, 0.5]
tableToPoints1Display.BackfaceRepresentation = 'Follow Frontface'
tableToPoints1Display.BackfaceAmbientColor = [1.0, 1.0, 1.0]
tableToPoints1Display.BackfaceDiffuseColor = [1.0, 1.0, 1.0]
tableToPoints1Display.BackfaceOpacity = 1.0
tableToPoints1Display.Position = [0.0, 0.0, 0.0]
tableToPoints1Display.Scale = [1.0, 1.0, 1.0]
tableToPoints1Display.Orientation = [0.0, 0.0, 0.0]
tableToPoints1Display.Origin = [0.0, 0.0, 0.0]
tableToPoints1Display.Pickable = 1
tableToPoints1Display.Texture = None
tableToPoints1Display.Triangulate = 0
tableToPoints1Display.NonlinearSubdivisionLevel = 1
tableToPoints1Display.UseDataPartitions = 0
tableToPoints1Display.OSPRayUseScaleArray = 0
tableToPoints1Display.OSPRayScaleArray = 'Broken Bonds'
tableToPoints1Display.OSPRayScaleFunction = 'PiecewiseFunction'
tableToPoints1Display.Orient = 0
tableToPoints1Display.OrientationMode = 'Direction'
tableToPoints1Display.SelectOrientationVectors = 'Broken Bonds'
tableToPoints1Display.Scaling = 0
tableToPoints1Display.ScaleMode = 'No Data Scaling Off'
tableToPoints1Display.ScaleFactor = 0.09975
tableToPoints1Display.SelectScaleArray = 'Broken Bonds'
tableToPoints1Display.GlyphType = 'Arrow'
tableToPoints1Display.SelectionCellLabelBold = 0
tableToPoints1Display.SelectionCellLabelColor = [0.0, 1.0, 0.0]
tableToPoints1Display.SelectionCellLabelFontFamily = 'Arial'
tableToPoints1Display.SelectionCellLabelFontSize = 18
tableToPoints1Display.SelectionCellLabelItalic = 0
tableToPoints1Display.SelectionCellLabelJustification = 'Left'
tableToPoints1Display.SelectionCellLabelOpacity = 1.0
tableToPoints1Display.SelectionCellLabelShadow = 0
tableToPoints1Display.SelectionPointLabelBold = 0
tableToPoints1Display.SelectionPointLabelColor = [1.0, 1.0, 0.0]
tableToPoints1Display.SelectionPointLabelFontFamily = 'Arial'
tableToPoints1Display.SelectionPointLabelFontSize = 18
tableToPoints1Display.SelectionPointLabelItalic = 0
tableToPoints1Display.SelectionPointLabelJustification = 'Left'
tableToPoints1Display.SelectionPointLabelOpacity = 1.0
tableToPoints1Display.SelectionPointLabelShadow = 0
tableToPoints1Display.PolarAxes = 'PolarAxesRepresentation'
tableToPoints1Display.GaussianRadius = 0.049875
tableToPoints1Display.ShaderPreset = 'Sphere'
tableToPoints1Display.Emissive = 0
tableToPoints1Display.ScaleByArray = 0
tableToPoints1Display.SetScaleArray = ['POINTS', 'Broken Bonds']
tableToPoints1Display.ScaleTransferFunction = 'PiecewiseFunction'
tableToPoints1Display.OpacityByArray = 0
tableToPoints1Display.OpacityArray = ['POINTS', 'Broken Bonds']
tableToPoints1Display.OpacityTransferFunction = 'PiecewiseFunction'

# init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
tableToPoints1Display.OSPRayScaleFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]

# init the 'Arrow' selected for 'GlyphType'
tableToPoints1Display.GlyphType.TipResolution = 6
tableToPoints1Display.GlyphType.TipRadius = 0.1
tableToPoints1Display.GlyphType.TipLength = 0.35
tableToPoints1Display.GlyphType.ShaftResolution = 6
tableToPoints1Display.GlyphType.ShaftRadius = 0.03
tableToPoints1Display.GlyphType.Invert = 0

# init the 'PolarAxesRepresentation' selected for 'PolarAxes'
tableToPoints1Display.PolarAxes.Visibility = 0
tableToPoints1Display.PolarAxes.Translation = [0.0, 0.0, 0.0]
tableToPoints1Display.PolarAxes.Scale = [1.0, 1.0, 1.0]
tableToPoints1Display.PolarAxes.Orientation = [0.0, 0.0, 0.0]
tableToPoints1Display.PolarAxes.EnableCustomBounds = [0, 0, 0]
tableToPoints1Display.PolarAxes.CustomBounds = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
tableToPoints1Display.PolarAxes.EnableCustomRange = 0
tableToPoints1Display.PolarAxes.CustomRange = [0.0, 1.0]
tableToPoints1Display.PolarAxes.PolarAxisVisibility = 1
tableToPoints1Display.PolarAxes.RadialAxesVisibility = 1
tableToPoints1Display.PolarAxes.DrawRadialGridlines = 1
tableToPoints1Display.PolarAxes.PolarArcsVisibility = 1
tableToPoints1Display.PolarAxes.DrawPolarArcsGridlines = 1
tableToPoints1Display.PolarAxes.NumberOfRadialAxes = 0
tableToPoints1Display.PolarAxes.AutoSubdividePolarAxis = 1
tableToPoints1Display.PolarAxes.NumberOfPolarAxis = 0
tableToPoints1Display.PolarAxes.MinimumRadius = 0.0
tableToPoints1Display.PolarAxes.MinimumAngle = 0.0
tableToPoints1Display.PolarAxes.MaximumAngle = 90.0
tableToPoints1Display.PolarAxes.RadialAxesOriginToPolarAxis = 1
tableToPoints1Display.PolarAxes.Ratio = 1.0
tableToPoints1Display.PolarAxes.PolarAxisColor = [1.0, 1.0, 1.0]
tableToPoints1Display.PolarAxes.PolarArcsColor = [1.0, 1.0, 1.0]
tableToPoints1Display.PolarAxes.LastRadialAxisColor = [1.0, 1.0, 1.0]
tableToPoints1Display.PolarAxes.SecondaryPolarArcsColor = [1.0, 1.0, 1.0]
tableToPoints1Display.PolarAxes.SecondaryRadialAxesColor = [1.0, 1.0, 1.0]
tableToPoints1Display.PolarAxes.PolarAxisTitleVisibility = 1
tableToPoints1Display.PolarAxes.PolarAxisTitle = 'Radial Distance'
tableToPoints1Display.PolarAxes.PolarAxisTitleLocation = 'Bottom'
tableToPoints1Display.PolarAxes.PolarLabelVisibility = 1
tableToPoints1Display.PolarAxes.PolarLabelFormat = '%-#6.3g'
tableToPoints1Display.PolarAxes.PolarLabelExponentLocation = 'Labels'
tableToPoints1Display.PolarAxes.RadialLabelVisibility = 1
tableToPoints1Display.PolarAxes.RadialLabelFormat = '%-#3.1f'
tableToPoints1Display.PolarAxes.RadialLabelLocation = 'Bottom'
tableToPoints1Display.PolarAxes.RadialUnitsVisibility = 1
tableToPoints1Display.PolarAxes.ScreenSize = 10.0
tableToPoints1Display.PolarAxes.PolarAxisTitleColor = [1.0, 1.0, 1.0]
tableToPoints1Display.PolarAxes.PolarAxisTitleOpacity = 1.0
tableToPoints1Display.PolarAxes.PolarAxisTitleFontFamily = 'Arial'
tableToPoints1Display.PolarAxes.PolarAxisTitleBold = 0
tableToPoints1Display.PolarAxes.PolarAxisTitleItalic = 0
tableToPoints1Display.PolarAxes.PolarAxisTitleShadow = 0
tableToPoints1Display.PolarAxes.PolarAxisTitleFontSize = 12
tableToPoints1Display.PolarAxes.PolarAxisLabelColor = [1.0, 1.0, 1.0]
tableToPoints1Display.PolarAxes.PolarAxisLabelOpacity = 1.0
tableToPoints1Display.PolarAxes.PolarAxisLabelFontFamily = 'Arial'
tableToPoints1Display.PolarAxes.PolarAxisLabelBold = 0
tableToPoints1Display.PolarAxes.PolarAxisLabelItalic = 0
tableToPoints1Display.PolarAxes.PolarAxisLabelShadow = 0
tableToPoints1Display.PolarAxes.PolarAxisLabelFontSize = 12
tableToPoints1Display.PolarAxes.LastRadialAxisTextColor = [1.0, 1.0, 1.0]
tableToPoints1Display.PolarAxes.LastRadialAxisTextOpacity = 1.0
tableToPoints1Display.PolarAxes.LastRadialAxisTextFontFamily = 'Arial'
tableToPoints1Display.PolarAxes.LastRadialAxisTextBold = 0
tableToPoints1Display.PolarAxes.LastRadialAxisTextItalic = 0
tableToPoints1Display.PolarAxes.LastRadialAxisTextShadow = 0
tableToPoints1Display.PolarAxes.LastRadialAxisTextFontSize = 12
tableToPoints1Display.PolarAxes.SecondaryRadialAxesTextColor = [1.0, 1.0, 1.0]
tableToPoints1Display.PolarAxes.SecondaryRadialAxesTextOpacity = 1.0
tableToPoints1Display.PolarAxes.SecondaryRadialAxesTextFontFamily = 'Arial'
tableToPoints1Display.PolarAxes.SecondaryRadialAxesTextBold = 0
tableToPoints1Display.PolarAxes.SecondaryRadialAxesTextItalic = 0
tableToPoints1Display.PolarAxes.SecondaryRadialAxesTextShadow = 0
tableToPoints1Display.PolarAxes.SecondaryRadialAxesTextFontSize = 12
tableToPoints1Display.PolarAxes.EnableDistanceLOD = 1
tableToPoints1Display.PolarAxes.DistanceLODThreshold = 0.7
tableToPoints1Display.PolarAxes.EnableViewAngleLOD = 1
tableToPoints1Display.PolarAxes.ViewAngleLODThreshold = 0.7
tableToPoints1Display.PolarAxes.SmallestVisiblePolarAngle = 0.5
tableToPoints1Display.PolarAxes.PolarTicksVisibility = 1
tableToPoints1Display.PolarAxes.ArcTicksOriginToPolarAxis = 1
tableToPoints1Display.PolarAxes.TickLocation = 'Both'
tableToPoints1Display.PolarAxes.AxisTickVisibility = 1
tableToPoints1Display.PolarAxes.AxisMinorTickVisibility = 0
tableToPoints1Display.PolarAxes.ArcTickVisibility = 1
tableToPoints1Display.PolarAxes.ArcMinorTickVisibility = 0
tableToPoints1Display.PolarAxes.DeltaAngleMajor = 10.0
tableToPoints1Display.PolarAxes.DeltaAngleMinor = 5.0
tableToPoints1Display.PolarAxes.PolarAxisMajorTickSize = 0.0
tableToPoints1Display.PolarAxes.PolarAxisTickRatioSize = 0.3
tableToPoints1Display.PolarAxes.PolarAxisMajorTickThickness = 1.0
tableToPoints1Display.PolarAxes.PolarAxisTickRatioThickness = 0.5
tableToPoints1Display.PolarAxes.LastRadialAxisMajorTickSize = 0.0
tableToPoints1Display.PolarAxes.LastRadialAxisTickRatioSize = 0.3
tableToPoints1Display.PolarAxes.LastRadialAxisMajorTickThickness = 1.0
tableToPoints1Display.PolarAxes.LastRadialAxisTickRatioThickness = 0.5
tableToPoints1Display.PolarAxes.ArcMajorTickSize = 0.0
tableToPoints1Display.PolarAxes.ArcTickRatioSize = 0.3
tableToPoints1Display.PolarAxes.ArcMajorTickThickness = 1.0
tableToPoints1Display.PolarAxes.ArcTickRatioThickness = 0.5
tableToPoints1Display.PolarAxes.Use2DMode = 0
tableToPoints1Display.PolarAxes.UseLogAxis = 0

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
tableToPoints1Display.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
tableToPoints1Display.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]

# reset view to fit data
renderView1.ResetCamera()
"""
#changing interaction mode based on data extents
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [0.49875, 0.49726838175, 10000.0]
renderView1.CameraFocalPoint = [0.49875, 0.49726838175, 0.0]

# create a new 'Glyph'
glyph1 = Glyph(Input=tableToPoints1,
    GlyphType='Box')
#glyph1.Scalars = ['POINTS', 'None']
#glyph1.Vectors = ['POINTS', 'None']
#glyph1.Orient = 0
#glyph1.ScaleMode = 'off'
glyph1.ScaleFactor = 0.0051
glyph1.GlyphMode = 'All Points'
#glyph1.MaximumNumberOfSamplePoints = 5000
#glyph1.Seed = 10339
#glyph1.Stride = 1
#glyph1.GlyphTransform = 'Transform2'
"""
# init the 'Arrow' selected for 'GlyphType'
glyph1.GlyphType.TipResolution = 6
glyph1.GlyphType.TipRadius = 0.1
glyph1.GlyphType.TipLength = 0.35
glyph1.GlyphType.ShaftResolution = 6
glyph1.GlyphType.ShaftRadius = 0.03
glyph1.GlyphType.Invert = 0

# init the 'Transform2' selected for 'GlyphTransform'
glyph1.GlyphTransform.Translate = [0.0, 0.0, 0.0]
glyph1.GlyphTransform.Rotate = [0.0, 0.0, 0.0]
glyph1.GlyphTransform.Scale = [1.0, 1.0, 1.0]

# Properties modified on glyph1
glyph1.GlyphType = 'Box'
glyph1.Scalars = ['POINTS', 'None']
glyph1.Orient = 0
glyph1.ScaleFactor = 0.0051
glyph1.GlyphMode = 'All Points'
"""
# show data in view
glyph1Display = Show(glyph1, renderView1)
# trace defaults for the display properties.
glyph1Display.Representation = 'Surface'
glyph1Display.AmbientColor = [1.0, 1.0, 1.0]
glyph1Display.ColorArrayName = [None, '']
glyph1Display.DiffuseColor = [1.0, 1.0, 1.0]
glyph1Display.LookupTable = None
glyph1Display.MapScalars = 1
glyph1Display.InterpolateScalarsBeforeMapping = 1
glyph1Display.Opacity = 1.0
glyph1Display.PointSize = 2.0
glyph1Display.LineWidth = 1.0
glyph1Display.Interpolation = 'Gouraud'
glyph1Display.Specular = 0.0
glyph1Display.SpecularColor = [1.0, 1.0, 1.0]
glyph1Display.SpecularPower = 100.0
glyph1Display.Ambient = 0.0
glyph1Display.Diffuse = 1.0
glyph1Display.EdgeColor = [0.0, 0.0, 0.5]
glyph1Display.BackfaceRepresentation = 'Follow Frontface'
glyph1Display.BackfaceAmbientColor = [1.0, 1.0, 1.0]
glyph1Display.BackfaceDiffuseColor = [1.0, 1.0, 1.0]
glyph1Display.BackfaceOpacity = 1.0
glyph1Display.Position = [0.0, 0.0, 0.0]
glyph1Display.Scale = [1.0, 1.0, 1.0]
glyph1Display.Orientation = [0.0, 0.0, 0.0]
glyph1Display.Origin = [0.0, 0.0, 0.0]
glyph1Display.Pickable = 1
glyph1Display.Texture = None
glyph1Display.Triangulate = 0
glyph1Display.NonlinearSubdivisionLevel = 1
glyph1Display.UseDataPartitions = 0
glyph1Display.OSPRayUseScaleArray = 0
glyph1Display.OSPRayScaleArray = 'Broken Bonds'
glyph1Display.OSPRayScaleFunction = 'PiecewiseFunction'
glyph1Display.Orient = 0
glyph1Display.OrientationMode = 'Direction'
glyph1Display.SelectOrientationVectors = 'Broken Bonds'
glyph1Display.Scaling = 0
glyph1Display.ScaleMode = 'No Data Scaling Off'
glyph1Display.ScaleFactor = 0.10025999487843365
glyph1Display.SelectScaleArray = 'Broken Bonds'
glyph1Display.GlyphType = 'Arrow'
glyph1Display.SelectionCellLabelBold = 0
glyph1Display.SelectionCellLabelColor = [0.0, 1.0, 0.0]
glyph1Display.SelectionCellLabelFontFamily = 'Arial'
glyph1Display.SelectionCellLabelFontSize = 18
glyph1Display.SelectionCellLabelItalic = 0
glyph1Display.SelectionCellLabelJustification = 'Left'
glyph1Display.SelectionCellLabelOpacity = 1.0
glyph1Display.SelectionCellLabelShadow = 0
glyph1Display.SelectionPointLabelBold = 0
glyph1Display.SelectionPointLabelColor = [1.0, 1.0, 0.0]
glyph1Display.SelectionPointLabelFontFamily = 'Arial'
glyph1Display.SelectionPointLabelFontSize = 18
glyph1Display.SelectionPointLabelItalic = 0
glyph1Display.SelectionPointLabelJustification = 'Left'
glyph1Display.SelectionPointLabelOpacity = 1.0
glyph1Display.SelectionPointLabelShadow = 0
glyph1Display.PolarAxes = 'PolarAxesRepresentation'
glyph1Display.GaussianRadius = 0.050129997439216825
glyph1Display.ShaderPreset = 'Sphere'
glyph1Display.Emissive = 0
glyph1Display.ScaleByArray = 0
glyph1Display.SetScaleArray = ['POINTS', 'Broken Bonds']
glyph1Display.ScaleTransferFunction = 'PiecewiseFunction'
glyph1Display.OpacityByArray = 0
glyph1Display.OpacityArray = ['POINTS', 'Broken Bonds']
glyph1Display.OpacityTransferFunction = 'PiecewiseFunction'

# init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
glyph1Display.OSPRayScaleFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]

# init the 'Arrow' selected for 'GlyphType'
glyph1Display.GlyphType.TipResolution = 6
glyph1Display.GlyphType.TipRadius = 0.1
glyph1Display.GlyphType.TipLength = 0.35
glyph1Display.GlyphType.ShaftResolution = 6
glyph1Display.GlyphType.ShaftRadius = 0.03
glyph1Display.GlyphType.Invert = 0

# init the 'PolarAxesRepresentation' selected for 'PolarAxes'
glyph1Display.PolarAxes.Visibility = 0
glyph1Display.PolarAxes.Translation = [0.0, 0.0, 0.0]
glyph1Display.PolarAxes.Scale = [1.0, 1.0, 1.0]
glyph1Display.PolarAxes.Orientation = [0.0, 0.0, 0.0]
glyph1Display.PolarAxes.EnableCustomBounds = [0, 0, 0]
glyph1Display.PolarAxes.CustomBounds = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
glyph1Display.PolarAxes.EnableCustomRange = 0
glyph1Display.PolarAxes.CustomRange = [0.0, 1.0]
glyph1Display.PolarAxes.PolarAxisVisibility = 1
glyph1Display.PolarAxes.RadialAxesVisibility = 1
glyph1Display.PolarAxes.DrawRadialGridlines = 1
glyph1Display.PolarAxes.PolarArcsVisibility = 1
glyph1Display.PolarAxes.DrawPolarArcsGridlines = 1
glyph1Display.PolarAxes.NumberOfRadialAxes = 0
glyph1Display.PolarAxes.AutoSubdividePolarAxis = 1
glyph1Display.PolarAxes.NumberOfPolarAxis = 0
glyph1Display.PolarAxes.MinimumRadius = 0.0
glyph1Display.PolarAxes.MinimumAngle = 0.0
glyph1Display.PolarAxes.MaximumAngle = 90.0
glyph1Display.PolarAxes.RadialAxesOriginToPolarAxis = 1
glyph1Display.PolarAxes.Ratio = 1.0
glyph1Display.PolarAxes.PolarAxisColor = [1.0, 1.0, 1.0]
glyph1Display.PolarAxes.PolarArcsColor = [1.0, 1.0, 1.0]
glyph1Display.PolarAxes.LastRadialAxisColor = [1.0, 1.0, 1.0]
glyph1Display.PolarAxes.SecondaryPolarArcsColor = [1.0, 1.0, 1.0]
glyph1Display.PolarAxes.SecondaryRadialAxesColor = [1.0, 1.0, 1.0]
glyph1Display.PolarAxes.PolarAxisTitleVisibility = 1
glyph1Display.PolarAxes.PolarAxisTitle = 'Radial Distance'
glyph1Display.PolarAxes.PolarAxisTitleLocation = 'Bottom'
glyph1Display.PolarAxes.PolarLabelVisibility = 1
glyph1Display.PolarAxes.PolarLabelFormat = '%-#6.3g'
glyph1Display.PolarAxes.PolarLabelExponentLocation = 'Labels'
glyph1Display.PolarAxes.RadialLabelVisibility = 1
glyph1Display.PolarAxes.RadialLabelFormat = '%-#3.1f'
glyph1Display.PolarAxes.RadialLabelLocation = 'Bottom'
glyph1Display.PolarAxes.RadialUnitsVisibility = 1
glyph1Display.PolarAxes.ScreenSize = 10.0
glyph1Display.PolarAxes.PolarAxisTitleColor = [1.0, 1.0, 1.0]
glyph1Display.PolarAxes.PolarAxisTitleOpacity = 1.0
glyph1Display.PolarAxes.PolarAxisTitleFontFamily = 'Arial'
glyph1Display.PolarAxes.PolarAxisTitleBold = 0
glyph1Display.PolarAxes.PolarAxisTitleItalic = 0
glyph1Display.PolarAxes.PolarAxisTitleShadow = 0
glyph1Display.PolarAxes.PolarAxisTitleFontSize = 12
glyph1Display.PolarAxes.PolarAxisLabelColor = [1.0, 1.0, 1.0]
glyph1Display.PolarAxes.PolarAxisLabelOpacity = 1.0
glyph1Display.PolarAxes.PolarAxisLabelFontFamily = 'Arial'
glyph1Display.PolarAxes.PolarAxisLabelBold = 0
glyph1Display.PolarAxes.PolarAxisLabelItalic = 0
glyph1Display.PolarAxes.PolarAxisLabelShadow = 0
glyph1Display.PolarAxes.PolarAxisLabelFontSize = 12
glyph1Display.PolarAxes.LastRadialAxisTextColor = [1.0, 1.0, 1.0]
glyph1Display.PolarAxes.LastRadialAxisTextOpacity = 1.0
glyph1Display.PolarAxes.LastRadialAxisTextFontFamily = 'Arial'
glyph1Display.PolarAxes.LastRadialAxisTextBold = 0
glyph1Display.PolarAxes.LastRadialAxisTextItalic = 0
glyph1Display.PolarAxes.LastRadialAxisTextShadow = 0
glyph1Display.PolarAxes.LastRadialAxisTextFontSize = 12
glyph1Display.PolarAxes.SecondaryRadialAxesTextColor = [1.0, 1.0, 1.0]
glyph1Display.PolarAxes.SecondaryRadialAxesTextOpacity = 1.0
glyph1Display.PolarAxes.SecondaryRadialAxesTextFontFamily = 'Arial'
glyph1Display.PolarAxes.SecondaryRadialAxesTextBold = 0
glyph1Display.PolarAxes.SecondaryRadialAxesTextItalic = 0
glyph1Display.PolarAxes.SecondaryRadialAxesTextShadow = 0
glyph1Display.PolarAxes.SecondaryRadialAxesTextFontSize = 12
glyph1Display.PolarAxes.EnableDistanceLOD = 1
glyph1Display.PolarAxes.DistanceLODThreshold = 0.7
glyph1Display.PolarAxes.EnableViewAngleLOD = 1
glyph1Display.PolarAxes.ViewAngleLODThreshold = 0.7
glyph1Display.PolarAxes.SmallestVisiblePolarAngle = 0.5
glyph1Display.PolarAxes.PolarTicksVisibility = 1
glyph1Display.PolarAxes.ArcTicksOriginToPolarAxis = 1
glyph1Display.PolarAxes.TickLocation = 'Both'
glyph1Display.PolarAxes.AxisTickVisibility = 1
glyph1Display.PolarAxes.AxisMinorTickVisibility = 0
glyph1Display.PolarAxes.ArcTickVisibility = 1
glyph1Display.PolarAxes.ArcMinorTickVisibility = 0
glyph1Display.PolarAxes.DeltaAngleMajor = 10.0
glyph1Display.PolarAxes.DeltaAngleMinor = 5.0
glyph1Display.PolarAxes.PolarAxisMajorTickSize = 0.0
glyph1Display.PolarAxes.PolarAxisTickRatioSize = 0.3
glyph1Display.PolarAxes.PolarAxisMajorTickThickness = 1.0
glyph1Display.PolarAxes.PolarAxisTickRatioThickness = 0.5
glyph1Display.PolarAxes.LastRadialAxisMajorTickSize = 0.0
glyph1Display.PolarAxes.LastRadialAxisTickRatioSize = 0.3
glyph1Display.PolarAxes.LastRadialAxisMajorTickThickness = 1.0
glyph1Display.PolarAxes.LastRadialAxisTickRatioThickness = 0.5
glyph1Display.PolarAxes.ArcMajorTickSize = 0.0
glyph1Display.PolarAxes.ArcTickRatioSize = 0.3
glyph1Display.PolarAxes.ArcMajorTickThickness = 1.0
glyph1Display.PolarAxes.ArcTickRatioThickness = 0.5
glyph1Display.PolarAxes.Use2DMode = 0
glyph1Display.PolarAxes.UseLogAxis = 0

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
glyph1Display.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
glyph1Display.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]

# hide data in view
Hide(tableToPoints1, renderView1)

# set scalar coloring
ColorBy(glyph1Display, ('POINTS', 'Pressure'))

# rescale color and/or opacity maps used to include current data range
glyph1Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
glyph1Display.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'Pressure'
pressureLUT = GetColorTransferFunction('Pressure')
#pressureLUT.LockDataRange = 0
pressureLUT.InterpretValuesAsCategories = 0
pressureLUT.ShowCategoricalColorsinDataRangeOnly = 0
pressureLUT.RescaleOnVisibilityChange = 0
pressureLUT.EnableOpacityMapping = 0
pressureLUT.RGBPoints = [1000000.0, 0.231373, 0.298039, 0.752941, 1000064.0, 0.865003, 0.865003, 0.865003, 1000128.0, 0.705882, 0.0156863, 0.14902]
pressureLUT.UseLogScale = 0
pressureLUT.ColorSpace = 'Diverging'
pressureLUT.UseBelowRangeColor = 0
pressureLUT.BelowRangeColor = [0.0, 0.0, 0.0]
pressureLUT.UseAboveRangeColor = 0
pressureLUT.AboveRangeColor = [1.0, 1.0, 1.0]
pressureLUT.NanColor = [1.0, 1.0, 0.0]
pressureLUT.Discretize = 1
pressureLUT.NumberOfTableValues = 256
pressureLUT.ScalarRangeInitialized = 1.0
pressureLUT.HSVWrap = 0
pressureLUT.VectorComponent = 0
pressureLUT.VectorMode = 'Magnitude'
pressureLUT.AllowDuplicateScalars = 1
pressureLUT.Annotations = []
pressureLUT.ActiveAnnotatedValues = []
pressureLUT.IndexedColors = []

# get color legend/bar for pressureLUT in view renderView1
pressureLUTColorBar = GetScalarBar(pressureLUT, renderView1)
pressureLUTColorBar.AutoOrient = 1
pressureLUTColorBar.Orientation = 'Vertical'
pressureLUTColorBar.Position = [0.85, 0.05]
pressureLUTColorBar.Position2 = [0.12, 0.43]
pressureLUTColorBar.Title = 'Pressure'
pressureLUTColorBar.ComponentTitle = ''
pressureLUTColorBar.TitleJustification = 'Centered'
pressureLUTColorBar.TitleColor = [0.0, 0.0, 0.0]
pressureLUTColorBar.TitleOpacity = 1.0
pressureLUTColorBar.TitleFontFamily = 'Arial'
pressureLUTColorBar.TitleBold = 0
pressureLUTColorBar.TitleItalic = 0
pressureLUTColorBar.TitleShadow = 0
pressureLUTColorBar.TitleFontSize = 7
pressureLUTColorBar.LabelColor = [0.0, 0.0, 0.0]
pressureLUTColorBar.LabelOpacity = 1.0
pressureLUTColorBar.LabelFontFamily = 'Arial'
pressureLUTColorBar.LabelBold = 0
pressureLUTColorBar.LabelItalic = 0
pressureLUTColorBar.LabelShadow = 0
pressureLUTColorBar.LabelFontSize = 7
pressureLUTColorBar.AutomaticLabelFormat = 1
pressureLUTColorBar.LabelFormat = '%-#6.3g'
pressureLUTColorBar.NumberOfLabels = 5
pressureLUTColorBar.DrawTickMarks = 1
pressureLUTColorBar.DrawSubTickMarks = 1
pressureLUTColorBar.DrawTickLabels = 1
pressureLUTColorBar.AddRangeLabels = 1
pressureLUTColorBar.RangeLabelFormat = '%4.3e'
pressureLUTColorBar.DrawAnnotations = 1
pressureLUTColorBar.AddRangeAnnotations = 0
pressureLUTColorBar.AutomaticAnnotations = 0
pressureLUTColorBar.DrawNanAnnotation = 0
pressureLUTColorBar.NanAnnotation = 'NaN'
pressureLUTColorBar.TextPosition = 'Ticks right/top, annotations left/bottom'
pressureLUTColorBar.AspectRatio = 20.0

# change scalar bar placement
pressureLUTColorBar.Position = [0.7419925512104283, 0.3055701179554391]
pressureLUTColorBar.Position2 = [0.12, 0.42999999999999994]

#### saving camera placements for all active views

# current camera placement for renderView1
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [0.49875, 0.49726838175, 10000.0]
renderView1.CameraFocalPoint = [0.49875, 0.49726838175, 0.0]
renderView1.CameraParallelScale = 0.7027673284880392

#### uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).
