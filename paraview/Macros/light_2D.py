from paraview.simple import *

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
