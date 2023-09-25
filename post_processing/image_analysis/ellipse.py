

import cv2
import numpy as np

# Load your image
image = cv2.imread('rose_norm_p_py_bb_008000_nx.png')

# Define the lower and upper bounds of the red color in BGR format
lower_red = np.array([0, 0, 100])
upper_red = np.array([100, 100, 255])

# Threshold the image to isolate the red region
binary_image = cv2.inRange(image, lower_red, upper_red)

# Calculate moments of the binary image
moments = cv2.moments(binary_image)

for key, value in moments.items():
    print(f'{key}: {value}')

# Calculate ellipse parameters
x_c = moments['m10'] / moments['m00']
y_c = moments['m01'] / moments['m00']

print(x_c,y_c)

cov_xx = moments['mu20'] / moments['m00'] - x_c**2
cov_yy = moments['mu02'] / moments['m00'] - y_c**2
cov_xy = moments['mu11'] / moments['m00'] - x_c * y_c

print(cov_xx,cov_yy,cov_xy)

print(2 * (cov_xx + cov_yy - np.sqrt((cov_xx - cov_yy)**2 + 4 * cov_xy**2)))

theta = 0.5 * np.arctan2(2 * cov_xy, cov_xx - cov_yy)

# Calculate major and minor axis lengths
a = np.sqrt(2 * (cov_xx + cov_yy + np.sqrt((cov_xx - cov_yy)**2 + 4 * cov_xy**2)))
b = np.sqrt(2 * (cov_xx + cov_yy - np.sqrt((cov_xx - cov_yy)**2 + 4 * cov_xy**2)))

# Draw the ellipse on the original image
ellipse_params = ((int(x_c), int(y_c)), (int(a), int(b)), np.degrees(theta))
cv2.ellipse(image, ellipse_params, (0, 255, 0), 2)

# Display the image with the fitted ellipse
cv2.imshow('Fitted Ellipse', image)
# Wait for a key press or for a timeout (e.g., 5000 milliseconds or 5 seconds)
key = cv2.waitKey(10000)

# Check if the key pressed is 'q' or 'Q' (or any other desired key)
if key in (ord('q'), ord('Q')):
    cv2.destroyAllWindows()
