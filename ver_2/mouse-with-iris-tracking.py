import cv2
import numpy as np
import dlib
import pyautogui
import os

def on_threshold(x):
    pass

# variable defenitions
screen = np.zeros((1080, 1920, 3), np.uint8)

#changes directory to current
os.chdir("/home/aryan/Documents/Python/EyeText/ver_2")
cap = cv2.VideoCapture(0)

# turns on video
cv2.namedWindow('frame')
# creates trackbar
cv2.createTrackbar('threshold', 'frame', 0, 255, on_threshold)

# blob detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

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
    global frame
    _, frame = cap.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    mask = np.zeros(frame.shape, dtype=np.uint8)
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
            global eye
            eye = frame[min_y-1: max_y, min_x : max_x]
            eye = cv2.resize(eye, None, fx=3, fy=3)
            gray_eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
            #gray_eye = cv2.GaussianBlur(gray_eye, (7, 7), 0)
            _, img = cv2.threshold(gray_eye, threshold, 255, cv2.THRESH_BINARY_INV)
            keypoints = blob_process(img, blob_detector)
            for keypoint in keypoints:
                s = keypoint.size
                x = keypoint.pt[0]
                y = keypoint.pt[1]
                #print(s, y, x)
                cx = int(x)
                cy = int(y)
                cv2.circle(eye, (cx, cy), 5, (0, 0, 255), -1)
                return(cx, cy)
            #print(keypoints)
            cv2.drawKeypoints(eye, keypoints, eye, (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.polylines(frame, [left_eye_region], True, (0,0,255), 2)
            #cv2.circle(screen, (0, 540), 6, (255, 255, 255), -1)

        # cv2.imshow('eye', eye)
        #cv2.imshow('mask', mask)
    # cv2.imshow('frame', frame)
    #cv2.imshow('screen', screen)

def callibrate():
    """
    cv2.namedWindow('up')
    cv2.resizeWindow('up', 200, 300)
    cv2.moveWindow('up', 860, 0)

    cv2.namedWindow('down')
    cv2.resizeWindow('down', 200, 300)
    cv2.moveWindow('down', 860, 1600)

    cv2.namedWindow('left')
    cv2.resizeWindow('left', 200, 300)
    cv2.moveWindow('left', 0, 390)

    cv2.namedWindow('right')
    cv2.resizeWindow('right', 200, 300)
    cv2.moveWindow('right', 1920, 390)
    """
    click = False
    eyeCoords = irisCoord()
    g = str(eyeCoords)
    cv2.setMouseCallback('screen', click_pos)
    if click == True:
        cv2.putText(screen, g, (20, 540), cv2.FONT_HERSHEY_COMPLEX, 7, (255, 255, 255), 2)
    #print(g)
    
# cv2.putText(screen, "NOT DETECTED, ADJUST PARAMETERS", (200, 540), cv2.FONT_HERSHEY_COMPLEX, 9, (255, 255, 255), 2)

    if str(eye) == '[]':
        pass
    else:
        cv2.imshow('eye', eye)
    cv2.imshow('screen', screen)
    cv2.resizeWindow('right', 200, 300)
    cv2.moveWindow('right', 1920, 390)

    eyeCoords = irisCoord()
    g = str(eyeCoords)
    cv2.setMouseCallback('screen', click_pos)
    #print(g)
    
# cv2.putText(screen, "NOT DETECTED, ADJUST PARAMETERS", (200, 540), cv2.FONT_HERSHEY_COMPLEX, 9, (255, 255, 255), 2)

    if str(eye) == '[]':
        pass
    else:
        cv2.imshow('eye', eye)
    cv2.imshow('screen', screen)
    cv2.imshow('frame', frame)

def main():
    while True:
        #irisCoord()
        callibrate()
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    
if __name__=='__main__':
    main()

cap.release()
cv2.destroyAllWindows()