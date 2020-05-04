# Standard imports
import cv2
import numpy as np
# Read image
im = cv2.imread("eye.jpeg")
# Set up the detector with default parameters.
detector = cv2.SimpleBlobDetector_create()
# Detect blobs.
_, img = cv2.threshold(im, 42, 255, cv2.THRESH_BINARY)
keypoints = detector.detect(img)
# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 
# Show keypoints
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)
