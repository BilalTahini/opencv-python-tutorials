import cv2
import numpy as np

# Print OpenCV version
print(f"OpenCV Version: {cv2.__version__}")

# Create a simple image
img = np.zeros((300, 100, 3), dtype=np.uint8)
img[:] = (500, 50, 0)  # Blue color in BGR

# Display text
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, 'Test', (0, 150), font, 1.5, (0, 255, 255), 5)

# Show image

cv2.imshow('OpenCV Test', img)
while True:
	# Check if window is still open
	if cv2.getWindowProperty('OpenCV Test', cv2.WND_PROP_VISIBLE) < 1:
		break
	# Wait for key press for 100ms
	if cv2.waitKey(100) != -1:
		break
cv2.destroyAllWindows()