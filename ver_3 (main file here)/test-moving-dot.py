import numpy as np
import cv2

win = np.zeros((300, 300, 3), np.uint8)
y = 1
f=1
while True:
    cv2.circle(win, (10, y), 3, (0, 0, 255), -1)
    if f%10 == 0:
        win[:] = 0
        f = 1
    y = y+1
    f = f+1
    cv2.imshow('win', win)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cv2.destroyAllWindows()