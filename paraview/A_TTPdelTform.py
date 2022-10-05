"""
1. table to points, from csv to point coordinates and properties
2. dalaunay filter to fill gaps between points
3. trainsform filter (set to 1, but ready to be edited)


External functions are defined in my_functions.py which must be imported

"""
# trace generated using paraview version 5.6.0

import sys
# add the path to directory with macros to the sys path
sys.path.append('/home/home01/scgf/.config/ParaView/Macros')  # change if path to A_TTPsphBclip.py is different

from my_functions import * 


LoadPalette(paletteName='WhiteBackground')

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


# update the view to ensure updated data information
renderView1.Update()

set_colormap_details(transform1Display)

# toggle 3D widget visibility (only when running from the GUI)
Hide3DWidgets(proxy=transform1.Transform)


# update the view to ensure updated data information
renderView1.Update()
