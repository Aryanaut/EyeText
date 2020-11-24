import cv2
import numpy as np 
import threading
import glob
from matplotlib import pyplot as plt

GSTREAMER_PIPELINE0 = 'nvarguscamerasrc sensor_id=0 ! video/x-raw(memory:NVMM), width=3280, height=2464, format=(string)NV12, framerate=21/1 ! nvvidconv flip-method=0 ! video/x-raw, width=960, height=616, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink'
GSTREAMER_PIPELINE1 = 'nvarguscamerasrc sensor_id=1 ! video/x-raw(memory:NVMM), width=3280, height=2464, format=(string)NV12, framerate=21/1 ! nvvidconv flip-method=0 ! video/x-raw, width=960, height=616, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink'
leftCamera = cv2.VideoCapture(GSTREAMER_PIPELINE1, cv2.CAP_GSTREAMER)
rightCamera = cv2.VideoCapture(GSTREAMER_PIPELINE0, cv2.CAP_GSTREAMER)

run = True

left = np.zeros((100, 100, 3), np.uint8)
right = np.zeros((100, 100, 3), np.uint8)

def left_vid():
    while True:
        global left, run
        _, left = leftCamera.read()
        left = cv2.flip(left, 0)
        left = cv2.resize(left, None, fx=0.5, fy=0.5)
        if run == False:
            break

def right_vid():
    while True:
        global right, run
        _, right = rightCamera.read()
        right = cv2.flip(right, 0)
        right = cv2.resize(right, None, fx=0.5, fy=0.5)
        if run == False:
            break

def main():
    l = threading.Thread(target=left_vid)
    l.start()
    r = threading.Thread(target=right_vid)
    r.start()
    global left, right, run
    while True:
        leftGray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        rightGray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        cv2.imshow('right', right)
        cv2.imshow('left', left)

        if cv2.waitKey(1) & 0xff == ord('q'):
            if leftGray.shape == rightGray.shape:
                stereo = cv2.StereoBM_create(numDisparities=48, blockSize=13)
                disparity = stereo.compute(leftGray, rightGray)
                plt.imshow(disparity, 'gray')
                plt.show()
            else:
                print("NOPE")
            break
            run = False
        else:
            run = True

    leftCamera.release()
    rightCamera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()