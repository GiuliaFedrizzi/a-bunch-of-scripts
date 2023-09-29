"""
1. loops through objects in the pipeline
2. applies table to points, using the function in A_TTPsphBclip.py
3. applies Delaunay filter
4. applies transform filter (no actual transform)


Needs my_functions.py, path to file is /home/home01/scgf/myscripts/paraview

"""

import sys
# add the path to directory with my_functions to the sys path
sys.path.append('/home/home01/scgf/myscripts/paraview')  # change if path to my_functions.py is different

from my_functions import * 


# get active view
renderView1 = GetActiveViewOrCreate('RenderView')


# prepare the view: palette, lighting, 2D view and update
LoadPalette(paletteName='WhiteBackground')

renderView1.Update()


for k,v in GetSources().items():
    """ loop that goes through every object in the pipeline. 
       GetSources() returns a dictionary of (name, id) object pairs. 
       Since multiple objects can have the same name, the (name,id) pair identifies objects uniquely."""
    SetActiveSource(v)   # sets the current active view to the first item in pipeline
    
    domain_size,found_scale = get_scale()  # domain_size is the sccale, found_scacle is 0 or 1

    applyTableToPointsSingle()
    delaunay2D1 = Delaunay2D()
    # create a new 'Transform'
    transform1 = Transform(Input=delaunay2D1)
    transform1.Transform = 'Transform'

    if found_scale:
        # apply the right numbers to transform
        transform1.Transform.Translate = [50.0-(domain_size/2), 100.0-domain_size, 0.0]
        transform1.Transform.Scale = [domain_size, domain_size, domain_size]
    else:
        print("No scale")

    Hide3DWidgets(proxy=transform1.Transform)  # hides the "box", equivalent to unticking "Show Box"
    print("k[0]: ",k[0])
    print("v: ",v)

# view the last one (last in the loop)
transform1Display = Show(transform1, renderView1)

#applyLightingAnd2Dview()

# Apply colormaps.
# it is enought to set them only once. This applies them to the last object in the loop
set_colormap_pressure(transform1Display)


set_colormap_broken_bonds(transform1Display)


applyLightingAnd2Dview()



# show one of the delaunays
#delaunay2D1Display = Show(delaunay2D1, renderView1)

