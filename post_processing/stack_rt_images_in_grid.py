"""
get all the images in rt0.*/images_in_grid/ according to their timestep 
and put them in one big image

goal: compare different relaxation thresholds
to be run from prt*, parent directory of rt0.*
"""


from PIL import Image, ImageDraw, ImageFont
import os
import glob

# List of rt directories
rt_dirs = sorted((glob.glob("rt0.*"))) 
# print(rt_dirs)

tsteps = list(range(1, 20, 1)) + list(range(20, 141, 5)) + list(range(150, 501, 20)) + list(range(500, 801, 20))  # 

for t in tsteps:
    timestep = "t"+str(t).zfill(3)
    print(timestep)

    # List to store the images
    images = []

    # Loop through each directory and open the image
    for rt_dir in rt_dirs:
        img_path = os.path.join(rt_dir, "images_in_grid", f"visc_mRate_160_{timestep}.png")
        if os.path.exists(img_path):
            img = Image.open(img_path)

            # Trim 1/6 of the top and 1/6 of the bottom
            width, height = img.size
            crop_top = height // 6
            crop_bottom = height - crop_top
            img_cropped = img.crop((0, crop_top, width, crop_bottom))
            # Add directory name in the top right corner
            draw = ImageDraw.Draw(img_cropped)
            # font = ImageFont.truetype("/Library/Fonts/Arial Unicode.ttf", size=80)
            font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", size=80)
            
            
            # font = ImageFont.load_default()  # or specify a custom font
            # text_color = (0, 255, 0)  # Green color
            text_color = (0, 0, 0)  
            text = rt_dir
            text_width, text_height = draw.textsize(text, font=font)
            position = (width - text_width - 10, 10)  # 10 pixels from the top right corner
            draw.text(position, text, fill=text_color, font=font)
            images.append(img_cropped)

    # Assuming all images are of the same size after cropping
    if images:
        width, height = images[0].size
        combined_height = height * len(images)

        # Create a new image with combined height
        combined_image = Image.new('RGB', (width, combined_height))

        # Paste each image
        y_offset = 0
        for img in images:
            combined_image.paste(img, (0, y_offset))
            y_offset += height

        # Save the combined image
        combined_image.save(os.getcwd().split("/")[-1]+"_"+timestep+".png")
        # combined_image.show()
