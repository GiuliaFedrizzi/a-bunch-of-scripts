# table to points, then change to Points visualisation


from paraview.simple import *
sys.path.append('/home/home01/scgf/myscripts/paraview')  # change if path to my_functions.py is different
from my_functions import * 


tableToPain = TableToPoints()
tableToPain.XColumn = 'x coord'
tableToPain.YColumn = 'y coord'
tableToPain.ZColumn = 'z coord'
tableToPain.a2DPoints = 1

tableToPoints1 = GetActiveSource()

renderView1 = GetActiveViewOrCreate('RenderView')


# get display properties
tableToPoints1Display = GetDisplayProperties(tableToPoints1, view=renderView1)


# change representation type
tableToPoints1Display.SetRepresentationType('Points')

# Properties modified on tableToPoints1Display
tableToPoints1Display.PointSize = 4.0


set_colormap_broken_bonds(tableToPoints1Display)
set_colormap_porosity(tableToPoints1Display)
set_colormap_pressure(tableToPoints1Display)
hide_colormaps()
