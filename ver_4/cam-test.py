import cv2
import numpy as np

GSTREAMER_PIPELINE0 = 'nvarguscamerasrc sensor_id=0 ! video/x-raw(memory:NVMM), width=3280, height=2464, format=(string)'
cap = cv2.VideoCapture(GSTREAMER_PIPELINE0, cv2.CAP_GSTREAMER)

while True:
    ret, frame = cap.read()

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()