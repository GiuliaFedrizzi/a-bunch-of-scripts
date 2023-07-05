from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

part_to_analyse = 'w'

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


crop_im = 1
if part_to_analyse == 'w': # whole domain
# Setting the points for cropped image
# left = 316; top = 147; right = 996; bottom = 819 # worked when images were generated on my laptop
    # left = 475; top = 223; right = 1490; bottom = 1228 # worked when images were generated on ARC
    # left = 475; top = 223; right = 1490; bottom = 1210 # latest version 5/7/23
    left = min(X)+5; top = min(Y)+7; right = max(X)-5; bottom = max(Y)-5 # auto from blue
    # left = 428; top = 162; right = 1492; bottom = 1211  # worked for threeAreas/prod/p01
elif part_to_analyse == 'b':
    left = 475; top = 1027; right = 1490; bottom = 1228 # BOTTOM - melt-production zone
elif part_to_analyse == 't':
    #left = 475; top = 223; right = 1490; bottom = 1027 # TOP = melt-production zone - if prod zone is 0-0.2
    #left = 475; top = 223; right = 1490; bottom = 1178 # TOP = melt-production zone  - if prod zone is 0-0.05:  1228-(1228-223)*0.05
    left = 475; top = 223; right = 1490; bottom = 1128 # TOP = melt-production zone  - if prod zone is 0-0.1
    
    
elif part_to_analyse == 'f': # full, or field = do not crop
    crop_im = 0  # in this case, do not crop 

if crop_im:  # crop the image if flag is true
    # Cropped image of above dimension
    im = im.crop((left, top, right, bottom))

im.save("cropped.png")
