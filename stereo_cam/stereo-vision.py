import cv2
import numpy as np
import matplotlib.pyplot as plt

main_img = cv2.imread('main.png', 0)
left_img = main_img[0:246, 0:331]
right_img = main_img[0:246, 332:663]

stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(left_img, right_img)
plt.imshow(disparity, 'gray')
plt.show()
cv2.imshow('left_img', left_img)
cv2.imshow('right_img', right_img)
cv2.waitKey(0)

cv2.destroyAllWindows()
