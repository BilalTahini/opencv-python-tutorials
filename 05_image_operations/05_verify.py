import cv2
import numpy as np

"""Create a sample image with shapes for demonstration"""
# Create a blank image with white background
img = np.ones((400, 600, 3), dtype=np.uint8) * 255

# Draw a blue rectangle
cv2.rectangle(img, (50, 50), (200, 150), (50, 30, 150), -1)

# Draw a green circle
cv2.circle(img, (400, 130), 100, (90, 20, 100), -1)

# Draw a red triangle
pts = np.array([[300, 300], [200, 200], [400, 200]], np.int32)
pts = pts.reshape((-1, 1, 2))
cv2.fillPoly(img, [pts], (60, 80, 120))

# Add some text
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, 'OpenCV', (250, 350), font, 2, (0, 0, 0), 3)

# Add some noise for filtering demonstrations
noise = np.zeros(img.shape, np.uint8)
cv2.randu(noise, 0, 50)
img = cv2.add(img, noise)

cv2.imwrite('image1.jpg', img)

"""Create a sample image with shapes for demonstration"""
# Create a blank image with white background
img = np.ones((400, 600, 3), dtype=np.uint8) * 255

# Draw a blue rectangle
cv2.rectangle(img, (70, 70), (100, 200), (100, 200, 50), -1)

# Draw a green circle
cv2.circle(img, (100, 300), 40, (0, 170, 50), -1)

# Draw a red triangle
pts = np.array([[300, 150], [200, 250], [400, 250]], np.int32)
pts = pts.reshape((-1, 1, 2))
cv2.fillPoly(img, [pts], (30, 0, 10))

# Add some text
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, 'OpenCV', (250, 350), font, 2, (0, 0, 0), 3)

# Add some noise for filtering demonstrations
noise = np.zeros(img.shape, np.uint8)
cv2.randu(noise, 0, 50)
img = cv2.add(img, noise)

cv2.imwrite('image2.jpg', img)

# Load two images
img1 = cv2.imread('image1.jpg')
img2 = cv2.imread('image2.jpg')

# Resize the second image to match the first if needed
img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

# Add the images
added_img = cv2.add(img1, img2)

# Display the result
cv2.imshow('Added Image', added_img)
cv2.waitKey(0)

# Blend the images with weights
# Parameters:
# - First image
# - Weight of the first image
# - Second image
# - Weight of the second image
# - Scalar added to each sum
alpha = 0.7  # Weight of the first image
beta = 0.3   # Weight of the second image
blended_img = cv2.addWeighted(img1, alpha, img2, beta, 0)

# Display the result
cv2.imshow('Blended Image', blended_img)
cv2.waitKey(0)

# Subtract img2 from img1
subtracted_img = cv2.subtract(img1, img2)

# Display the result
cv2.imshow('Subtracted Image', subtracted_img)
cv2.waitKey(0)

# Multiply the images
multiplied_img = cv2.multiply(img1, img2)

# Display the result
cv2.imshow('Multiplied Image', multiplied_img)
cv2.waitKey(0)

# Divide img1 by img2
divided_img = cv2.divide(img1, img2)

# Display the result
cv2.imshow('Divided Image', divided_img)
cv2.waitKey(0)

# Create a black image
height, width = img1.shape[:2]
mask = np.zeros((height, width), dtype=np.uint8)

# Draw a white circle in the middle
center = (width // 2, height // 2)
radius = 100
cv2.circle(mask, center, radius, 255, -1)

# Display the mask
cv2.imshow('Mask', mask)
cv2.waitKey(0)

# Apply the mask using bitwise AND
# Parameters:
# - First image
# - Second image
masked_img = cv2.bitwise_and(img1, img1, mask=mask)

# Display the result
cv2.imshow('Masked Image (AND)', masked_img)
cv2.waitKey(0)

# Create another image with a different shape
img3 = np.zeros((height, width, 3), dtype=np.uint8)
cv2.rectangle(img3, (width//4, height//4), (3*width//4, 3*height//4), (0, 0, 255), -1)

# Apply bitwise OR
or_img = cv2.bitwise_or(img1, img3)

# Display the result
cv2.imshow('OR Operation', or_img)
cv2.waitKey(0)

# Apply bitwise XOR
xor_img = cv2.bitwise_xor(img1, img3)

# Display the result
cv2.imshow('XOR Operation', xor_img)
cv2.waitKey(0)

# Apply bitwise NOT
not_img = cv2.bitwise_not(img1)

# Display the result
cv2.imshow('NOT Operation', not_img)
cv2.waitKey(0)

# Create a mask with a gradient
gradient_mask = np.zeros((height, width), dtype=np.uint8)
for i in range(width):
    gradient_mask[:, i] = i * 255 // width

# Blend images using the gradient mask
img1_masked = cv2.bitwise_and(img1, img1, mask=gradient_mask)
cv2.imshow('Gradient Mask', gradient_mask)
cv2.waitKey(0)

img2_masked = cv2.bitwise_and(img2, img2, mask=cv2.bitwise_not(gradient_mask))
cv2.imshow('Inverse Gradient Mask', cv2.bitwise_not(gradient_mask))
cv2.waitKey(0)

blended_with_mask = cv2.add(img1_masked, img2_masked)

# Display the result
cv2.imshow('Blended with Mask', blended_with_mask)
cv2.waitKey(0)

# Assume we have a foreground mask (e.g., from segmentation)
foreground_mask = np.zeros((height, width), dtype=np.uint8)
cv2.rectangle(foreground_mask, (width//4, height//4), (3*width//4, 3*height//4), 255, -1)
cv2.imshow('Foreground Mask', foreground_mask)
cv2.waitKey(0)

# Extract the foreground
foreground = cv2.bitwise_and(img1, img1, mask=foreground_mask)
cv2.imshow('Foreground', foreground)
cv2.waitKey(0)

# Create a colored background
background = np.ones((height, width, 3), dtype=np.uint8) * [0, 255, 0]  # Green background
background_mask = cv2.bitwise_not(foreground_mask)
background = cv2.bitwise_and(background, background, mask=background_mask)
background = background.astype(np.uint8)  # Ensure uint8 type for imshow

cv2.imshow('Background', background)
cv2.waitKey(0)

# Combine foreground and new background
result = cv2.add(foreground, background)

# Display the result
cv2.imshow('Background Removal', result)
cv2.waitKey(0)

# Load or create a logo
logo = np.zeros((100, 100, 3), dtype=np.uint8)
cv2.circle(logo, (50, 50), 40, (0, 0, 255), -1)
cv2.putText(logo, 'CV', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# Resize the logo if needed
logo_resized = cv2.resize(logo, (width//4, height//4))
logo_height, logo_width = logo_resized.shape[:2]

# Create a region of interest (ROI) in the top-right corner
roi = img1[0:logo_height, width-logo_width:width]

# Create a mask of the logo and its inverse
logo_gray = cv2.cvtColor(logo_resized, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(logo_gray, 10, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)

# Black-out the area of the logo in ROI
roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

# Take only the logo region from the logo image
logo_fg = cv2.bitwise_and(logo_resized, logo_resized, mask=mask)

# Put the logo in ROI and modify the original image
dst = cv2.add(roi_bg, logo_fg)
img1[0:logo_height, width-logo_width:width] = dst

# Display the result
cv2.imshow('Watermarked Image', img1)
cv2.waitKey(0)