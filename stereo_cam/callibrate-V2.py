import cv2
import numpy as np
import argparse
import threading

ap = argparse.ArgumentParser()
ap.add_argument('-s', '--stream', type=str, required=True, 
                    help='choose picam or webcam')
args = vars(ap.parse_args())

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((6*7, 3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
left = np.zeros((100, 100, 3), np.uint8)
right = np.zeros((100, 100, 3), np.uint8)

objpoints = []
imgpoints = []

streamType = args['stream']
if streamType == 'picam':
    GSTREAMER_PIPELINE0 = 'nvarguscamerasrc sensor_id=0 ! video/x-raw(memory:NVMM), width=3280, height=2464, format=(string)NV12, framerate=21/1 ! nvvidconv flip-method=0 ! video/x-raw, width=960, height=616, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink'
    GSTREAMER_PIPELINE1 = 'nvarguscamerasrc sensor_id=1 ! video/x-raw(memory:NVMM), width=3280, height=2464, format=(string)NV12, framerate=21/1 ! nvvidconv flip-method=0 ! video/x-raw, width=960, height=616, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink'
    leftCamera = cv2.VideoCapture(GSTREAMER_PIPELINE1, cv2.CAP_GSTREAMER)
    rightCamera = cv2.VideoCapture(GSTREAMER_PIPELINE0, cv2.CAP_GSTREAMER)
else:
    leftCamera = cv2.VideoCapture(1)
    rightCamera = cv2.VideoCapture(0)

def left_vid():
    while True:
        global left 
        _, left = leftCamera.read()
        left = cv2.flip(left, 0)

def right_vid():
    while True:
        global right
        _, right = rightCamera.read()
        right = cv2.flip(right, 0)

def main():
    l = threading.Thread(target=left_vid)
    l.start()
    r = threading.Thread(target=right_vid)
    r.start()
    global right, left
    while True:
        leftGray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        rightGray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

        cv2.imshow('right', right)
        cv2.imshow('left', left)

        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    leftCamera.release()
    rightCamera.release()

if __name__ == '__main__':
    main()