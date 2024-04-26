from PIL import Image
import numpy as np
from skimage.morphology import opening, closing, square
from skimage.filters import threshold_otsu

# Load the image
image_path = 'py_bb_016000_cropped.png'
original_image = Image.open(image_path)
image_array = np.array(original_image)

# Assuming white pixels (fractures) are 1 and blue pixels (background) are 0
# Invert the image if needed
fractures = np.invert(image_array[:, :, 0] > 128)

# Apply a binary threshold to clean up the image
thresh = threshold_otsu(fractures)
binary = fractures > thresh

# img_binary = Image.fromarray(binary)
# img_binary.save('py_bb_016000_cropped_binary.png')
# print("saved binary")

# Apply morphological opening to remove small objects
square_size = 4
selem = square(square_size)
opened = opening(binary, selem)
img_opened = Image.fromarray(opened)
img_opened.save('py_bb_016000_cropped_opened'+str(square_size)+'.png')
print("saved opened")

# Apply morphological closing to close small holes
closed = closing(opened, selem)

# Cast the processed image array to uint8 before creating the image
# processed_image_uint8 = np.invert(closed).astype(np.uint8) * 255

# Convert back to an image
processed_image = Image.fromarray(closed)

# Save the processed image
processed_image_path = 'py_bb_016000_cropped_processed.png'
processed_image.save(processed_image_path)


