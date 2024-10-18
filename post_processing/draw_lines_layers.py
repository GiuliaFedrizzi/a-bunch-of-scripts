from PIL import Image
import matplotlib.pyplot as plt

# Load the image
image_path = 'viz_bb_009000.png'
image = Image.open(image_path)

# Get image dimensions
width, height = image.size

# Calculate the position for the horizontal line (0.1428 of the height from the bottom)
line_y1 = height * (1 - 0.1428)
line_y2 = height * (1 - 0.2857)
line_y3 = height * (1 - 0.428)
line_y4 = height * (1 - 0.57)
line_y5 = height * (1 - 0.714)
line_y6 = height * (1 - 0.857)

# Plot the image
plt.imshow(image)

# Draw the horizontal line
plt.axhline(y=line_y1, color='red', linewidth=2)  # Line at 0.1428 of height from the bottom
plt.axhline(y=line_y2, color='red', linewidth=2)  # Line at 0.1428 of height from the bottom
plt.axhline(y=line_y3, color='red', linewidth=2)  # Line at 0.1428 of height from the bottom
plt.axhline(y=line_y4, color='red', linewidth=2)  # Line at 0.1428 of height from the bottom
plt.axhline(y=line_y5, color='red', linewidth=2)  # Line at 0.1428 of height from the bottom
plt.axhline(y=line_y6, color='red', linewidth=2)  # Line at 0.1428 of height from the bottom


# Show the plot with the lines
plt.axis('off')  # Turn off axes
plt.show()
