import cv2
import numpy as np
import os 

d = os.getcwd()
os.system('cd '+d)

kernel = np.array([[0, -1, 0], 
                   [-1, 5,-1], 
                   [0, -1, 0]])

img = cv2.imread('test2.jpg')
# img = cv2.filter2D(img, -1, kernel)
cv2.namedWindow('img')
gray_eye = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thr = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY_INV)
thr = cv2.dilate(thr, None, iterations=10)
thr = cv2.erode(thr, None, iterations=6)
thr = cv2.blur(thr, (12, 12), 0)
# thr = cv2.dilate(thr, None, iterations=9)

edged = cv2.Canny(thr, 79, 200)
circles = cv2.HoughCircles(edged,cv2.HOUGH_GRADIENT,1,20,
                            param1=50,param2=30,minRadius=0)

if circles is not None:
	circles = np.round(circles[0, :]).astype("int")
	for (x, y, r) in circles:
		cv2.circle(img, (x, y), r, (0, 255, 0), 4)
		cv2.rectangle(img, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

cv2.imshow('thr', thr)
cv2.imshow('edged', edged)
cv2.imshow('img', img)
cv2.imwrite('thr.png', thr)
cv2.waitKey(0)
cv2.destroyAllWindows()