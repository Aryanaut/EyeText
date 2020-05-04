import cv2
import numpy as np
import dlib
import os

#directory changing
cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

threshold = 42

detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.maxArea = 3000
blob_detector = cv2.SimpleBlobDetector_create(detector_params)

def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)
    
def blob_process(img, detection):
    keypoints = detection.detect(img)
    return keypoints
    

while True:
    _, frame = cap.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
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

        eye = frame[min_y: max_y+5, min_x-4 : max_x-4]
        eye = cv2.resize(eye, None, fx=5, fy=5)
        gray_eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
        _, img = cv2.threshold(gray_eye, threshold, 255, cv2.THRESH_BINARY)
        img = cv2.erode(img, None, iterations=2) #1
        #img = cv2.dilate(img, None, iterations=4) #2
        img = cv2.medianBlur(img, 5)
        keypoints = blob_process(img, blob_detector)
        print(keypoints)
        cv2.drawKeypoints(img, keypoints, eye, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        cv2.polylines(frame, [left_eye_region], True, (0,0,255), 2)

    cv2.imshow('eye', eye)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()