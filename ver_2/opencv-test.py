import cv2
import numpy as np

cap = cv2.VideoCapture(0)
screen = np.zeros((800, 800, 3), np.uint8)

while True:
    re, frame = cap.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
    screen[0:480, 0:640] = frame

    cv2.imshow('screen', screen)
    cv2.imshow('frame', frame)
    print(type(frame[0, 0][0]))
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()