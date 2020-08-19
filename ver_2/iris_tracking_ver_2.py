import cv2
import numpy as np
import dlib
import pyautogui
import os

def on_threshold(x):
    pass

#directory changing
#os.chdir("/home/aryan/Documents/EyeText/ver_2")
cap = cv2.VideoCapture(0)
#cap.set(cv2.CAP_PROP_FPS, 60)
cv2.namedWindow('frame')
cv2.createTrackbar('threshold', 'frame', 0, 255, on_threshold)
screen = np.zeros((1080, 1920, 3), np.uint8)


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#threshold = 42  

detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByColor = True
detector_params.blobColor = 255
#detector_params.filterByArea = True
#detector_params.maxArea = 3000
blob_detector = cv2.SimpleBlobDetector_create(detector_params)

def blob_process(img, detection):
    img = cv2.erode(img, None, iterations=2)
    img = cv2.dilate(img, None, iterations=30)
    img = cv2.medianBlur(img, 5)
    keypoints = detection.detect(img)
    return keypoints

def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)
    
while True:
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

            eye = frame[min_y-1: max_y, min_x : max_x]
            eye = cv2.resize(eye, None, fx=3, fy=3)
            gray_eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
            #gray_eye = cv2.GaussianBlur(gray_eye, (7, 7), 0)
            _, img = cv2.threshold(gray_eye, threshold, 255, cv2.THRESH_BINARY_INV)
            keypoints = blob_process(img, blob_detector)
            coords = np.column_stack(np.where(img > 0))
            """
            for coord in coords:
                mask[coord[0], coord[1]] = (0,0,255)
            print(coords)
            #cv2.circle(eye, (cX, cY), 6, (0, 0, 255), -1)
            """
            
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
            cv2.circle(screen, (0, 540), 6, (255, 255, 255), -1)

        cv2.imshow('eye', eye)
        #cv2.imshow('mask', mask)
    cv2.imshow('frame', frame)
    #cv2.imshow('screen', screen)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
