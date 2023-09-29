"""
1. table to points, from csv to point coordinates and properties
2. dalaunay filter to fill gaps between points
3. trainsform filter (set to 1, but ready to be edited)


External functions are defined in my_functions.py which must be imported
path is /home/home01/scgf/myscripts/paraview

"""
# trace generated using paraview version 5.6.0

import sys
# add the path to directory with macros to the sys path
sys.path.append('/home/home01/scgf/myscripts/paraview')  # change if path to my_functions.py is different

from my_functions import * 

domain_size,found_scale = get_scale()  # domain_size is the sccale, found_scacle is 0 or 1

LoadPalette(paletteName='WhiteBackground')

applyTableToPointsSingle()

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')

# call function that sets background and light
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

if found_scale:
    # apply the right numbers to transform
    transform1.Transform.Translate = [50.0-(domain_size/2), 100.0-domain_size, 0.0]
    transform1.Transform.Scale = [domain_size, domain_size, domain_size]
else:
  print("No scale")

# show data in view
transform1Display = Show(transform1, renderView1)

# hide data in view
Hide(delaunay2D1, renderView1)


# update the view to ensure updated data information
renderView1.Update()


set_colormap_pressure(transform1Display)
set_colormap_pressure(transform1Display)
set_colormap_broken_bonds(transform1Display)
set_colormap_healing(transform1Display)

brokenBondsLUT = GetColorTransferFunction('BrokenBonds')
pressureLUT = GetColorTransferFunction('Pressure')
porosityLUT = GetColorTransferFunction('Porosity')
healingLUT = GetColorTransferFunction('Healing')

HideScalarBarIfNotNeeded(pressureLUT, renderView1)
HideScalarBarIfNotNeeded(porosityLUT, renderView1)
HideScalarBarIfNotNeeded(healingLUT, renderView1)

# toggle 3D widget visibility (only when running from the GUI)
Hide3DWidgets(proxy=transform1.Transform)


# update the view to ensure updated data information
renderView1.Update()
