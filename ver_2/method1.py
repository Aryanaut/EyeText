import cv2
import numpy as np
import dlib
import pyautogui
import os
import time

def on_threshold(x):
    pass

click = False
# variable defenitions
screenW, screenH = pyautogui.size()
midptX = int(screenW / 2)
midptY = int(screenH / 2)
screen = np.zeros((screenH, screenW, 3), np.uint8)
error_msg = cv2.imread('/home/aryan/Documents/Python/EyeText/ver_2/err-msg.png')

#changes directory to current
os.chdir("/home/aryan/Documents/Python/EyeText/ver_2")
cap = cv2.VideoCapture(0)

cv2.namedWindow('frame')
# creates trackbar
cv2.createTrackbar('threshold', 'frame', 0, 255, on_threshold)

# blob detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# callibration points
points = {
        1 : (0, 0),
        2 : (0, 0),
        3 : (0, 0),
        4 : (0, 0),
    }

global cx, cy
cx, cy = 0, 0

global point
point = 1

global callibrationStatus
callibrationStatus = False

detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByColor = True
detector_params.blobColor = 255
#detector_params.filterByArea = True
#detector_params.maxArea = 3000
blob_detector = cv2.SimpleBlobDetector_create(detector_params)

# defines the process that looks for blobs
def blob_process(img, detection):
    img = cv2.erode(img, None, iterations=2)
    img = cv2.dilate(img, None, iterations=30)
    img = cv2.medianBlur(img, 5)
    keypoints = detection.detect(img)
    return keypoints

# defines function to identify when the mouse is clicked
def click_pos(event, x, y, flags, params):
    global click
    if event == cv2.EVENT_LBUTTONDOWN:
        click = True
    else:
        click = False

def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

# function to find the iris
def irisCoord():
    _, frame = cap.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
    # frame = cv2.resize(frame, (100, 100))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    mask = np.zeros(frame.shape, dtype=np.uint8)
    eye = np.zeros((200, 300, 3), np.uint8)
    #print(faces)
    rectangles = str(faces)
    #print(rectangles, type(rectangles))
    if rectangles == 'rectangles[]':
        #print('waiting...')
        pass
    else:
        for face in faces:
            x, y = face.left(), face.top()
            x1, y1 = face.right(), face.bottom()
            cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)
            landmarks = predictor(gray, face)

            left_point = (landmarks.part(36).x, landmarks.part(36).y)
            right_point = (landmarks.part(39).x, landmarks.part(39).y)
            top_center = midpoint(landmarks.part(37), landmarks.part(38))
            bottom_center = midpoint(landmarks.part(41), landmarks.part(40))

            #Gaze Ratio
            left_eye_region = np.array([(landmarks.part(36).x, landmarks.part(36).y),
                                (landmarks.part(37).x, landmarks.part(37).y),
                                (landmarks.part(38).x, landmarks.part(38).y),
                                (landmarks.part(39).x, landmarks.part(39).y),
                                (landmarks.part(40).x, landmarks.part(40).y),
                                (landmarks.part(41).x, landmarks.part(41).y)], np.int32)

            min_x = np.min(left_eye_region[:, 0])
            max_x = np.max(left_eye_region[:, 0])
            min_y = np.min(left_eye_region[:, 1])
            max_y = np.max(left_eye_region[:, 1])

            threshold = cv2.getTrackbarPos('threshold', 'frame')

            eye = frame[min_y-1: max_y, min_x : max_x]
            eye = cv2.resize(eye, None, fx=3, fy=3)
            # eye = cv2.resize(eye, (100, 100))
            gray_eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
            #gray_eye = cv2.GaussianBlur(gray_eye, (7, 7), 0)
            _, img = cv2.threshold(gray_eye, threshold, 255, cv2.THRESH_BINARY_INV)
            keypoints = blob_process(img, blob_detector)
            global cx, cy
            for keypoint in keypoints:
                s = keypoint.size
                x = keypoint.pt[0]
                y = keypoint.pt[1]
                #print(s, y, x)
                cx = int(x)
                cy = int(y)
                cv2.circle(eye, (cx, cy), 5, (0, 0, 255), -1)

            #print(keypoints)
            cv2.drawKeypoints(eye, keypoints, eye, (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.polylines(frame, [left_eye_region], True, (0,0,255), 2)
            #cv2.circle(screen, (0, 540), 6, (255, 255, 255), -1)

    return (eye, frame)

def callibrate():
    coordinates = (cx, cy)
    g = str((cx, cy))
    global point
    global callibrationStatus
    cv2.putText(screen, "UP", (900, 60), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 2)
    cv2.putText(screen, "RIGHT", (1700, 540), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 2)
    cv2.putText(screen, "LEFT", (20, 540), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 2)
    cv2.putText(screen, "DOWN", (860, 1000), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 2)

    if click == True:
            points[point] = coordinates
            point = point+1
            c = 20
            cv2.putText(screen, "ok", (c, 20), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            c = c+100
            if g == '(0, 0)':
                pass
                # cv2.putText(screen, "NOT DETECTED, ADJUST PARAMETERS", (200, 540), cv2.FONT_HERSHEY_COMPLEX, 9, (255, 255, 255), 2)
                """
                cv2.imshow('error', error_msg)
                time.sleep(2)
                cv2.destroyWindow('error')
                """
            if point > 4:
                cv2.putText(screen, "DONE", (200, 540), cv2.FONT_HERSHEY_COMPLEX, 9, (255, 255, 255), 2)
                callibrationStatus = True

    cv2.setMouseCallback('screen', click_pos)

def tracking():
    eye, frame = irisCoord()
    callibrate()
    global g
    if points[2] != (1000, 1000):
        if callibrationStatus == True:
            eyeCoordinates = points[2]
            eyeX = eyeCoordinates[0]
            eyeY = eyeCoordinates[1]
            if (eyeX, eyeY) != (0, 0):
                xRatio = int(eyeX/960)
                yRatio = int(eyeY/540)
                gazeCoords = (int(float(xRatio*cx)), int(float(yRatio*cy)))
                print(gazeCoords)
                cv2.circle(screen, gazeCoords, 3, (0, 0, 255), -1)

    else:
        pass
    screen[midptY-240:midptY+240, midptX-320:midptX+320] = frame
    # stacked = np.stack((frame, eye))
    # cv2.imshow('frame', frame)
    # cv2.imshow('frame', frame)
    cv2.imshow('windowEye', eye)
    cv2.imshow('screen', screen)

def main():
    while True:
        #irisCoord()
        tracking()
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

if __name__=='__main__':
    main()

cap.release()
cv2.destroyAllWindows()
print(points) 