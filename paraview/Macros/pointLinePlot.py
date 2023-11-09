"""
plot the vertical profile of the variable "Sigma_1" 
in the middle of the domain, x = 0.5
"""


# trace generated using paraview version 5.6.0
#
# To ensure correct image size when batch processing, please search 
# for and uncomment the line `# renderView*.ViewSize = [*,*]`

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# find source
tableToPoints1 = FindSource('TableToPoints1')

# create a new 'Point Line Interpolator'
pointLineInterpolator1 = PointLineInterpolator(Input=tableToPoints1,
    Source='High Resolution Line Source')
pointLineInterpolator1.Kernel = 'VoronoiKernel'
pointLineInterpolator1.Locator = 'Static Point Locator'

# # init the 'High Resolution Line Source' selected for 'Source'
# pointLineInterpolator1.Source.Point1 = [0.0, 0.00216165, 0.0]
# pointLineInterpolator1.Source.Point2 = [0.9975, 0.992229, 0.0]

# # find source
# th69visc_4_1e4vis1e4_mR_01 = FindSource('th69/visc_4_1e4/vis1e4_mR_01')

# # Create a new 'SpreadSheet View'
# spreadSheetView1 = CreateView('SpreadSheetView')
# spreadSheetView1.ColumnToSort = ''
# spreadSheetView1.BlockSize = 1024L
# # uncomment following to set a specific view size
# # spreadSheetView1.ViewSize = [400, 400]

# # get layout
layout1 = GetLayout()

# # place view in the layout
# layout1.AssignView(2, spreadSheetView1)

# # show data in view
# th69visc_4_1e4vis1e4_mR_01Display = Show(th69visc_4_1e4vis1e4_mR_01, spreadSheetView1)

# Properties modified on pointLineInterpolator1.Source
pointLineInterpolator1.Source.Point1 = [0.5, 0.00216165, 0.0]
pointLineInterpolator1.Source.Point2 = [0.5, 0.992229, 0.0]

# Create a new 'Line Chart View'
lineChartView1 = CreateView('XYChartView')
# lineChartView1.ViewSize = [772, 384]
# lineChartView1.ChartTitleFontFile = ''
# lineChartView1.LeftAxisTitleFontFile = ''
# lineChartView1.LeftAxisRangeMaximum = 6.66
# lineChartView1.LeftAxisLabelFontFile = ''
# lineChartView1.BottomAxisTitleFontFile = ''
# lineChartView1.BottomAxisRangeMaximum = 6.66
# lineChartView1.BottomAxisLabelFontFile = ''
# lineChartView1.RightAxisRangeMaximum = 6.66
# lineChartView1.RightAxisLabelFontFile = ''
# lineChartView1.TopAxisTitleFontFile = ''
# lineChartView1.TopAxisRangeMaximum = 6.66
# lineChartView1.TopAxisLabelFontFile = ''

# place view in the layout
# layout1.AssignView(6, lineChartView1)

# show data in view
pointLineInterpolator1Display = Show(pointLineInterpolator1, lineChartView1)

# trace defaults for the display properties.
pointLineInterpolator1Display.CompositeDataSetIndex = [0]
pointLineInterpolator1Display.XArrayName = 'Sigma_1'
pointLineInterpolator1Display.SeriesVisibility = [ 'Sigma_1']
pointLineInterpolator1Display.SeriesLabel = ['Sigma_1']
pointLineInterpolator1Display.SeriesColor = ['Sigma_1', '1', '0.5', '0']
# pointLineInterpolator1Display.SeriesPlotCorner = ['Actual Movement', '0', 'area_par_fluid', '0', 'Broken Bonds', '0', 'Differential Stress', '0', 'F_P_x', '0', 'F_P_y', '0', 'Fracture normal stress', '0', 'Fracture shear stress', '0', 'Fracture Time', '0', 'Fractures', '0', 'id', '0', 'Mean Stress', '0', 'Movement in Gravity', '0', 'Original Movement', '0', 'Permeability', '0', 'Porosity', '0', 'Pressure', '0', 'rad_par_fluid', '0', 'real_radius', '0', 'Sigma_1', '0', 'Sigma_2', '0', 'x velocity', '0', 'xy_melt_point', '0', 'y velocity', '0', 'Youngs Modulus', '0', 'Youngs Modulus E', '0', 'Youngs Modulus Par', '0', 'Youngs Modulus Real', '0', 'Points_X', '0', 'Points_Y', '0', 'Points_Z', '0', 'Points_Magnitude', '0']
pointLineInterpolator1Display.SeriesLabelPrefix = ''
pointLineInterpolator1Display.SeriesLineStyle = ['Sigma_1', '1']
pointLineInterpolator1Display.SeriesLineThickness = ['Sigma_1', '2']
pointLineInterpolator1Display.SeriesMarkerStyle = ['Sigma_1', '0']

# update the view to ensure updated data information
# spreadSheetView1.Update()

# update the view to ensure updated data information
lineChartView1.Update()

# # get color transfer function/color map for 'Sigma_1'
# sigma_1LUT = GetColorTransferFunction('Sigma_1')

# # Rescale transfer function
# sigma_1LUT.RescaleTransferFunction(-60903000.0, -57510100.0)

# # get opacity transfer function/opacity map for 'Sigma_1'
# sigma_1PWF = GetOpacityTransferFunction('Sigma_1')

# # Rescale transfer function
# sigma_1PWF.RescaleTransferFunction(-60903000.0, -57510100.0)

# Properties modified on pointLineInterpolator1Display
# pointLineInterpolator1Display.SeriesVisibility = []
# pointLineInterpolator1Display.SeriesColor = ['Actual Movement', '0', '0', '0', 'area_par_fluid', '0.889998', '0.100008', '0.110002', 'Broken Bonds', '0.220005', '0.489998', '0.719997', 'Differential Stress', '0.300008', '0.689998', '0.289998', 'F_P_x', '0.6', '0.310002', '0.639994', 'F_P_y', '1', '0.500008', '0', 'Fracture normal stress', '0.650004', '0.340002', '0.160006', 'Fracture shear stress', '0', '0', '0', 'Fracture Time', '0.889998', '0.100008', '0.110002', 'Fractures', '0.220005', '0.489998', '0.719997', 'id', '0.300008', '0.689998', '0.289998', 'Mean Stress', '0.6', '0.310002', '0.639994', 'Movement in Gravity', '1', '0.500008', '0', 'Original Movement', '0.650004', '0.340002', '0.160006', 'Permeability', '0', '0', '0', 'Porosity', '0.889998', '0.100008', '0.110002', 'Pressure', '0.220005', '0.489998', '0.719997', 'rad_par_fluid', '0.300008', '0.689998', '0.289998', 'real_radius', '0.6', '0.310002', '0.639994', 'Sigma_1', '1', '0.500008', '0', 'Sigma_2', '0.650004', '0.340002', '0.160006', 'x velocity', '0', '0', '0', 'xy_melt_point', '0.889998', '0.100008', '0.110002', 'y velocity', '0.220005', '0.489998', '0.719997', 'Youngs Modulus', '0.300008', '0.689998', '0.289998', 'Youngs Modulus E', '0.6', '0.310002', '0.639994', 'Youngs Modulus Par', '1', '0.500008', '0', 'Youngs Modulus Real', '0.650004', '0.340002', '0.160006', 'Points_X', '0', '0', '0', 'Points_Y', '0.889998', '0.100008', '0.110002', 'Points_Z', '0.220005', '0.489998', '0.719997', 'Points_Magnitude', '0.300008', '0.689998', '0.289998']
# pointLineInterpolator1Display.SeriesPlotCorner = ['Actual Movement', '0', 'Broken Bonds', '0', 'Differential Stress', '0', 'F_P_x', '0', 'F_P_y', '0', 'Fracture Time', '0', 'Fracture normal stress', '0', 'Fracture shear stress', '0', 'Fractures', '0', 'Mean Stress', '0', 'Movement in Gravity', '0', 'Original Movement', '0', 'Permeability', '0', 'Points_Magnitude', '0', 'Points_X', '0', 'Points_Y', '0', 'Points_Z', '0', 'Porosity', '0', 'Pressure', '0', 'Sigma_1', '0', 'Sigma_2', '0', 'Youngs Modulus', '0', 'Youngs Modulus E', '0', 'Youngs Modulus Par', '0', 'Youngs Modulus Real', '0', 'area_par_fluid', '0', 'id', '0', 'rad_par_fluid', '0', 'real_radius', '0', 'x velocity', '0', 'xy_melt_point', '0', 'y velocity', '0']
# pointLineInterpolator1Display.SeriesLineStyle = ['Actual Movement', '1', 'Broken Bonds', '1', 'Differential Stress', '1', 'F_P_x', '1', 'F_P_y', '1', 'Fracture Time', '1', 'Fracture normal stress', '1', 'Fracture shear stress', '1', 'Fractures', '1', 'Mean Stress', '1', 'Movement in Gravity', '1', 'Original Movement', '1', 'Permeability', '1', 'Points_Magnitude', '1', 'Points_X', '1', 'Points_Y', '1', 'Points_Z', '1', 'Porosity', '1', 'Pressure', '1', 'Sigma_1', '1', 'Sigma_2', '1', 'Youngs Modulus', '1', 'Youngs Modulus E', '1', 'Youngs Modulus Par', '1', 'Youngs Modulus Real', '1', 'area_par_fluid', '1', 'id', '1', 'rad_par_fluid', '1', 'real_radius', '1', 'x velocity', '1', 'xy_melt_point', '1', 'y velocity', '1']
# pointLineInterpolator1Display.SeriesLineThickness = ['Actual Movement', '2', 'Broken Bonds', '2', 'Differential Stress', '2', 'F_P_x', '2', 'F_P_y', '2', 'Fracture Time', '2', 'Fracture normal stress', '2', 'Fracture shear stress', '2', 'Fractures', '2', 'Mean Stress', '2', 'Movement in Gravity', '2', 'Original Movement', '2', 'Permeability', '2', 'Points_Magnitude', '2', 'Points_X', '2', 'Points_Y', '2', 'Points_Z', '2', 'Porosity', '2', 'Pressure', '2', 'Sigma_1', '2', 'Sigma_2', '2', 'Youngs Modulus', '2', 'Youngs Modulus E', '2', 'Youngs Modulus Par', '2', 'Youngs Modulus Real', '2', 'area_par_fluid', '2', 'id', '2', 'rad_par_fluid', '2', 'real_radius', '2', 'x velocity', '2', 'xy_melt_point', '2', 'y velocity', '2']
# pointLineInterpolator1Display.SeriesMarkerStyle = ['Actual Movement', '0', 'Broken Bonds', '0', 'Differential Stress', '0', 'F_P_x', '0', 'F_P_y', '0', 'Fracture Time', '0', 'Fracture normal stress', '0', 'Fracture shear stress', '0', 'Fractures', '0', 'Mean Stress', '0', 'Movement in Gravity', '0', 'Original Movement', '0', 'Permeability', '0', 'Points_Magnitude', '0', 'Points_X', '0', 'Points_Y', '0', 'Points_Z', '0', 'Porosity', '0', 'Pressure', '0', 'Sigma_1', '0', 'Sigma_2', '0', 'Youngs Modulus', '0', 'Youngs Modulus E', '0', 'Youngs Modulus Par', '0', 'Youngs Modulus Real', '0', 'area_par_fluid', '0', 'id', '0', 'rad_par_fluid', '0', 'real_radius', '0', 'x velocity', '0', 'xy_melt_point', '0', 'y velocity', '0']

# Properties modified on pointLineInterpolator1Display
# pointLineInterpolator1Display.SeriesVisibility = ['Sigma_1']

#### uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).