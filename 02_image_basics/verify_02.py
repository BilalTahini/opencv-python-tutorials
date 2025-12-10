#!/usr/bin/env python3
import cv2
import numpy as np
import os
import sys
# Read an image
# Parameters:
# - Path to the image file
# - Flag specifying how to read the image:
#   cv2.IMREAD_COLOR (1): Loads a color image (default)
#   cv2.IMREAD_GRAYSCALE (0): Loads image in grayscale mode
#   cv2.IMREAD_UNCHANGED (-1): Loads image as is including alpha channel
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'image.jpg')
img = cv2.imread(file_path, cv2.IMREAD_COLOR)

# Check if image was loaded successfully
if img is None:
    print("Error: Could not read image")
else:
    print("Image loaded successfully")

# Create a window and display the image
# Parameters:
# - Window name (string)
# - Image to be shown
cv2.imshow('Image Window', img)

while True:
    # Check if window is still open
    if cv2.getWindowProperty('Image Window', cv2.WND_PROP_VISIBLE) < 1:
        break
    # Wait for key press for 100ms
    if cv2.waitKey(100) != -1:
        break

#############################################

# Image dimensions (height, width, channels)
print(type(img.shape))
height, width, channels = img.shape
print(f"Image Dimensions: {width}x{height}")
print(f"Number of Channels: {channels}")

# Image data type
print(f"Image Data Type: {img.dtype}")

# Total number of pixels
print(f"Total Pixels: {img.size}")

##############################################

# Access a pixel value at row=100, col=50
# Returns [B, G, R] for color images
pixel = img[100, 50]
print(f"Pixel at (50, 100): {pixel}")

# Modify a pixel value
img[50:100, 50:100] = [255, 0, 0]  # Set to blue in BGR

# Access only blue channel of a pixel
blue = img[100, 50, 0]

# Modify only the green channel
img[200:300, 100:200, 1] = 100
print(f"value of changed pixel at 230, 130: {img[230, 130]}")
print(f"value of changed pixel at 70, 75: {img[70, 75]}")
cv2.imshow('Image Window', img)

while True:
    # Check if window is still open
    if cv2.getWindowProperty('Image Window', cv2.WND_PROP_VISIBLE) < 1:
        break
    # Wait for key press for 100ms
    if cv2.waitKey(100) != -1:
        break

##############################################

# Split the BGR image into separate channels
b, g, r = cv2.split(img)

# Display individual channels
cv2.imshow('Blue Channel', b)
cv2.imshow('Green Channel', g)
cv2.imshow('Red Channel', r)
cv2.waitKey(0)

# Merge channels back together
merged_img = cv2.merge((b, g, r))

################################################

# Resize to specific dimensions
resized_img = cv2.resize(img, (800, 600))

# Resize by scaling factor
# fx and fy are scaling factors for width and height
half_size = cv2.resize(img, None, fx=0.5, fy=0.5)
double_size = cv2.resize(img, None, fx=2, fy=2)

# Specify interpolation method
# cv2.INTER_AREA - good for shrinking
# cv2.INTER_CUBIC, cv2.INTER_LINEAR - good for enlarging
resized_img = cv2.resize(img, (800, 600), interpolation=cv2.INTER_CUBIC)

################################################

# Get the image dimensions
height, width = img.shape[:2]

# Define the rotation matrix
# Parameters: center point, angle (degrees), scale
rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), 45, 1)

# Apply the rotation
rotated_img = cv2.warpAffine(img, rotation_matrix, (width, height))
cv2.imshow('Rotated Image', rotated_img)
cv2.waitKey(0)