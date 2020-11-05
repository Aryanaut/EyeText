import cv2
import numpy as np
import os 

d = os.getcwd()
os.system('cd '+d)

kernel = np.array([[0, -1, 0], 
                   [-1, 5,-1], 
                   [0, -1, 0]])

img = cv2.imread('eye.png')
img = cv2.filter2D(img, -1, kernel)
cv2.namedWindow('img')
gray_eye = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thr = cv2.threshold(gray_eye, 80, 255, cv2.THRESH_BINARY_INV)
thr = cv2.dilate(thr, None, iterations=2)
thr = cv2.erode(thr, None, iterations=2)
# thr = cv2.dilate(thr, None, iterations=9)

edged = cv2.Canny(thr, 100, 200)
cont, h = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(img, cont, -1, (0, 255, 0), 3)

cv2.imshow('thr', thr)
cv2.imshow('edged', edged)
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()