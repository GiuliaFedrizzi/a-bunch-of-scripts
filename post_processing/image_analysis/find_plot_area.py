from PIL import Image
import numpy as np
# import matplotlib.pyplot as plt

part_to_analyse = 't'

# im = Image.open("py_bb_087000.png")

# Load image, ensure not palettised, and make into Numpy array
pim = Image.open('py_bb_087000.png').convert('RGB')
im_array  = np.array(pim)

# Define the blue colour we want to find - PIL uses RGB ordering
blue = [0,0,255]

# Get X and Y coordinates of all blue pixels
Y, X = np.where(np.all(im_array==blue,axis=2))

print(X,Y)
print(min(X),max(X),min(Y),max(Y))

im = Image.open("py_bb_087000.png")

left = min(X)+5; top = min(Y)+7; right = max(X)-5; bottom = max(Y)-5 # auto from blue
height = bottom - top


crop_im = 1
# if part_to_analyse == 'w': # whole domain
# Setting the points for cropped image


if part_to_analyse == 'b':
    top = top + 0.9*height # BOTTOM - melt-production zone
elif part_to_analyse == 't':
    # it is 0.9 of the original domain height
    bottom = bottom - 0.1*height # TOP = melt-production zone  - if prod zone is 0-0.1
    
    
elif part_to_analyse == 'f': # full, or field = do not crop
    crop_im = 0  # in this case, do not crop 

if crop_im:  # crop the image if flag is true
    # Cropped image of above dimension
    im = im.crop((left, top, right, bottom))

im.save("cropped.png")
