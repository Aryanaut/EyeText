import cv2
import numpy as np 
import threading
import glob
import argparse
from matplotlib import pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument('-t', '--streamType', type=str, required=True,
                help='Choose between -picam- and -webcam-')
args = vars(ap.parse_args())

streamType = args['streamType']

if streamType == 'picam':
    GSTREAMER_PIPELINE0 = 'nvarguscamerasrc sensor_id=0 ! video/x-raw(memory:NVMM), width=3280, height=2464, format=(string)NV12, framerate=21/1 ! nvvidconv flip-method=0 ! video/x-raw, width=960, height=616, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink'
    GSTREAMER_PIPELINE1 = 'nvarguscamerasrc sensor_id=1 ! video/x-raw(memory:NVMM), width=3280, height=2464, format=(string)NV12, framerate=21/1 ! nvvidconv flip-method=0 ! video/x-raw, width=960, height=616, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink'
    leftCamera = cv2.VideoCapture(GSTREAMER_PIPELINE1, cv2.CAP_GSTREAMER)
    rightCamera = cv2.VideoCapture(GSTREAMER_PIPELINE0, cv2.CAP_GSTREAMER)
else:
    leftCamera = cv2.VideoCapture(1)
    rightCamera = cv2.VideoCapture(0)

count = 0

left = np.zeros((100, 100, 3), np.uint8)
right = np.zeros((100, 100, 3), np.uint8)

def left_vid():
    while True:
        global left
        _, left = leftCamera.read()
        left = cv2.flip(left, 0)
        left = cv2.resize(left, None, fx=0.5, fy=0.5)

def right_vid():
    while True:
        global right
        _, right = rightCamera.read()
        right = cv2.flip(right, 0)
        right = cv2.resize(right, None, fx=0.5, fy=0.5)

def main():
    l = threading.Thread(target=left_vid)
    l.start()
    r = threading.Thread(target=right_vid)
    r.start()
    global left, right, count
    while True:
        try:
            leftGray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
            rightGray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
            cv2.putText(left, str(count), (20, 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
            cv2.putText(left, str(count), (20, 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
            cv2.imshow('right', right)
            cv2.imshow('left', left)
            right_string_name = 'right_im'+str(count)+'.png'
            left_string_name = 'left_im'+str(count)+'.png'

            if cv2.waitKey(1) & 0xff == ord(' '):
                try:
                    cv2.imwrite('images/'+right_string_name, rightGray)
                    cv2.imwrite('images/'+left_string_name, leftGray)
                    count += 1
                    print(count)
                except leftGray.shape == rightGray.shape:
                    print('Try again')

            if cv2.waitKey(1) & 0xff == ord('q'):
                break

        except TypeError:
            break
        except SyntaxError:
            break
        except EnvironmentError:
            break
        except UnboundLocalError:
            break

    leftCamera.release()
    rightCamera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
