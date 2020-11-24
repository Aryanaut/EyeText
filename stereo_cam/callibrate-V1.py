import cv2
import numpy as np 
import glob
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-t', '--stream', type=str, required=True, 
                help='choose picamera or webcam')
args = vars(ap.parse_args())

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((6*7, 3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

objpoints = []
imgpoints = []

streamType = args['stream']
if streamType == 'picam':
    GSTREAMER_PIPELINE = 'nvarguscamerasrc sensor_id=0 ! video/x-raw(memory:NVMM), width=3280, height=2464, format=(string)NV12, framerate=21/1 ! nvvidconv flip-method=0 ! video/x-raw, width=960, height=616, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink'
    cap = cv2.VideoCapture(GSTREAMER_PIPELINE, cv2.CAP_GSTREAMER)
else:
    cap = cv2.VideoCapture(0)

def main():
    while True:
        _, frame = cap.read()
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
        frame = cv2.flip(frame, 0)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)

        if ret == True:
            objpoints.append(objp)
            corners0 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners0)

            cv2.drawChessboardCorners(frame, (7, 6), corners0, ret)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()