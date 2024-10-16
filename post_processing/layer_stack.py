"""
get all the images in lr41, lr42l etc according to their timestep 
and put them in one big image, combined horizontally (def rates were combined vertically)

goal: compare different melt rate contrasts
to be run from layers, parent directory of lr*
"""


from PIL import Image, ImageDraw, ImageFont
import os
import glob

from useful_functions import getSaveFreq

def extract_numeric_value(directory_name):
    return float(directory_name[3:])

# List of def directories
layer_dirs = ['lr41','lr42','lr43']
os.chdir(layer_dirs[0])
save_freq = int(getSaveFreq())
os.chdir("..")

print(f'layer_dirs {layer_dirs}')
tsteps = list(range(24,35,1)) +  list(range(35, 110, 5)) + list(range(110, 310, 10))# + list(range(225, 420, 25)) 
# tsteps = list(range(24,25,1)) 
tsteps = [i*save_freq for i in tsteps] 

for t in tsteps:
    timestep = "t"+str(t).zfill(3)
    print(timestep)

    # List to store the images
    images = []

    # Loop through each directory and open the image
    for layer_dir in layer_dirs:
        img_path = os.path.join(layer_dir, f"{layer_dir}_{timestep}.png")
        if os.path.exists(img_path):
            print(f'img_path {img_path}')
            img = Image.open(img_path)
            width, height = img.size
            # no need to crop but keeping the name for now
            img_cropped = img
            # Add directory name in the top right corner
            draw = ImageDraw.Draw(img_cropped)
            font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", size=100)
            text_color = (0, 0, 0)  
            text = layer_dir
            text_width, text_height = draw.textsize(text, font=font)
            position = (text_width*2, 10)  # position of text: on the left
            draw.text(position, text, fill=text_color, font=font)
            images.append(img_cropped)
            
        else:
            print(f'no file {img_path}')
    # Assuming all images are of the same size after cropping
    if images:
        width, height = images[0].size
        combined_width = width * len(images)

        # Create a new image with combined width
        combined_image = Image.new('RGB', (combined_width, height))

        # Paste each image
        x_offset = 0
        for img in images:
            combined_image.paste(img, (x_offset, 0))
            x_offset += width

        # Save the combined image
        im_name = '_'.join(layer_dirs)
        combined_image.save("layer_images/"+im_name+"_"+timestep+".png")
        # combined_image.show()
    
