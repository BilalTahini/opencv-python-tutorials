import cv2
import numpy as np

# Read an image (in BGR format by default)
img = cv2.imread('image.jpg')
#cv2.imshow('Original Image', img)
#cv2.waitKey(0)
# 
print(img[20,20])
print(f'shape',img.shape)
# Convert BGR to Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(gray[20,20])
# Convert BGR to HSV (Hue, Saturation, Value)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
print(hsv[20,20])
# Convert BGR to RGB
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(rgb[20,20])
# Convert BGR to LAB
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
print(lab[20,20])

# Create a translation matrix
# tx = shift in x direction, ty = shift in y direction
tx, ty = 100, 50
translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])

# Apply the translation
# Parameters: image, translation matrix, output image dimensions
height, width = img.shape[:2]
translated = cv2.warpAffine(img, translation_matrix, (width, height))
print(f'shape',translated.shape)
# cv2.imshow('Translated Image', translated)
# cv2.waitKey(0)

# Define the rotation center, angle, and scale
center = (width // 2, height // 2)
angle = 45
scale = 1.5

# Calculate the rotation matrix
rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

# Apply the rotation
rotated = cv2.warpAffine(img, rotation_matrix, (width, height))

# cv2.imshow('Rotated Image', rotated)
# cv2.waitKey(0)

# Define three points from the input image
src_points = np.float32([[0, 0], [width - 1, 0], [0, height - 1]])

# Define where those points will be in the output image
dst_points = np.float32([[0, 0], [width - 1, 0], [width // 3, height - 1]])

# Calculate the affine transformation matrix
affine_matrix = cv2.getAffineTransform(src_points, dst_points)

# Apply the affine transformation
affine = cv2.warpAffine(img, affine_matrix, (width, height))

# cv2.imshow('Affine Transformed Image', affine)
# cv2.waitKey(0)

# Define four points in the input image
src_points = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])

# Define where those points will be in the output image
dst_points = np.float32([[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]])

# Calculate the perspective transformation matrix
perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

# Apply the perspective transformation
perspective = cv2.warpPerspective(img, perspective_matrix, (width, height))

# cv2.imshow('Perspective Transformed Image', perspective)
# cv2.waitKey(0)

# Apply average blur with a 5x5 kernel
blur = cv2.blur(img, (2, 2))

# cv2.imshow('Blurred Image', blur)
# cv2.waitKey(0)

# Apply Gaussian blur with a 5x5 kernel and standard deviation of 0
gaussian = cv2.GaussianBlur(img, (5, 5), 0)
# cv2.imshow('Gaussian Blurred Image', gaussian)
# cv2.waitKey(0)

# Apply median blur with a kernel size of 5
median = cv2.medianBlur(img, 5)
# cv2.imshow('Median Blurred Image', median)
# cv2.waitKey(0)

# Apply bilateral filter
# Parameters: image, diameter of pixel neighborhood, sigma color, sigma space
bilateral = cv2.bilateralFilter(img, 50, 75, 75)
# cv2.imshow('Bilateral Filtered Image', bilateral)
# cv2.waitKey(0)

# Define a kernel
kernel = np.ones((5, 5), np.uint8)

# Apply erosion
erosion = cv2.erode(img, kernel, iterations=1)

cv2.imshow('Eroded Image', erosion)
cv2.waitKey(0)

# Apply dilation
dilation = cv2.dilate(img, kernel, iterations=1)

cv2.imshow('Dilated Image', dilation)
cv2.waitKey(0)

# Apply opening
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

cv2.imshow('Opening Image', opening)
cv2.waitKey(0)

# Apply closing
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

cv2.imshow('Closing Image', closing)
cv2.waitKey(0)

# Apply morphological gradient
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

cv2.imshow('Morphological Gradient Image', gradient)
cv2.waitKey(0)

# Apply Laplacian
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
cv2.imshow('Laplacian Image', laplacian)
cv2.waitKey(0)

# Create a Gaussian pyramid (downsampling)
lower_res = cv2.pyrDown(img)  # Reduces to half the size
even_lower_res = cv2.pyrDown(lower_res)  # Reduces to quarter the size
cv2.imshow('Lower Resolution Image', lower_res)
cv2.waitKey(0)

# Create a Laplacian pyramid
higher_res = cv2.pyrUp(lower_res)  # Increases size but loses detail
laplacian = cv2.subtract(img, higher_res)  # Contains the lost detail
cv2.imshow('Laplacian Pyramid Image', laplacian)
cv2.waitKey(0)