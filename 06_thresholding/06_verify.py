import cv2
import numpy as np

# Read the image in grayscale
img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Apply simple thresholding
# Parameters:
# - source image (must be grayscale)
# - threshold value
# - maximum value to use with the THRESH_BINARY and THRESH_BINARY_INV thresholding types
# - thresholding type
cv2.imshow('Original Image', img)
cv2.waitKey(0)

ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
cv2.imshow('THRESH_BINARY', thresh1)
cv2.waitKey(0)
ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
cv2.imshow('THRESH_BINARY_INV', thresh2)
cv2.waitKey(0)
ret, thresh3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
cv2.imshow('THRESH_TRUNC', thresh3)
cv2.waitKey(0)
ret, thresh4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
cv2.imshow('THRESH_TOZERO', thresh4)
cv2.waitKey(0)
ret, thresh5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)
cv2.imshow('THRESH_TOZERO_INV', thresh5)
cv2.waitKey(0)

# Apply adaptive thresholding
# Parameters:
# - source image (must be grayscale)
# - maximum value to use with the THRESH_BINARY and THRESH_BINARY_INV thresholding types
# - adaptive method (ADAPTIVE_THRESH_MEAN_C or ADAPTIVE_THRESH_GAUSSIAN_C)
# - thresholding type (usually THRESH_BINARY or THRESH_BINARY_INV)
# - block size (size of the pixel neighborhood used to calculate the threshold)
# - constant subtracted from the mean or weighted mean
adaptive_thresh1 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                        cv2.THRESH_BINARY, 11, 2)
cv2.imshow('ADAPTIVE_THRESH_MEAN_C', adaptive_thresh1)
cv2.waitKey(0)

adaptive_thresh2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)
cv2.imshow('ADAPTIVE_THRESH_GAUSSIAN_C', adaptive_thresh2)
cv2.waitKey(0)

# Apply Otsu's thresholding
# Combine simple thresholding with Otsu's method using the additional flag THRESH_OTSU
ret, otsu_thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# The threshold value is returned as 'ret'
print(f"Otsu's threshold value: {ret}")
cv2.imshow("Otsu's Thresholding", otsu_thresh)
cv2.waitKey(0)

# Example of multi-level thresholding
ret1, thresh1 = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
cv2.imshow('Multi-level Thresholding - Level 1', thresh1)
cv2.waitKey(0)
ret2, thresh2 = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
cv2.imshow('Multi-level Thresholding - Level 2', thresh2)
cv2.waitKey(0)
ret3, thresh3 = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
cv2.imshow('Multi-level Thresholding - Level 3', thresh3)
cv2.waitKey(0)
ret4, thresh4 = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
cv2.imshow('Multi-level Thresholding - Level 4', thresh4)
cv2.waitKey(0)

# Combine the results
multi_level = np.zeros_like(img)
cv2.imshow('Multi-level Combined Before', multi_level)
cv2.waitKey(0)
multi_level = np.where(thresh1 == 255, 64, multi_level)
cv2.imshow('Multi-level Combined After Level 1', multi_level)
cv2.waitKey(0)
multi_level = np.where(thresh2 == 255, 128, multi_level)
cv2.imshow('Multi-level Combined After Level 2', multi_level)
cv2.waitKey(0)
multi_level = np.where(thresh3 == 255, 192, multi_level)
cv2.imshow('Multi-level Combined After Level 3', multi_level)
cv2.waitKey(0)
multi_level = np.where(thresh4 == 255, 255, multi_level)
cv2.imshow('Multi-level Combined Final', multi_level)
cv2.waitKey(0)