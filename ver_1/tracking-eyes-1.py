import cv2
import numpy as np
import dlib

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def midpoint(p1, p2):
    return int((p1.x+p2.x)/2), int((p1.y+p2.y)/2)

font = cv2.FONT_HERSHEY_PLAIN

def get_gaze_ratio(eye_points, facial_landmarks):
    left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                            (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                            (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                            (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                            (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                            (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)
    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [left_eye_region], True, 255, 2)
    cv2.fillPoly(mask, [left_eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)

    min_x = np.min(left_eye_region[:, 0])
    max_x = np.max(left_eye_region[:, 0])
    min_y = np.min(left_eye_region[:, 1])
    max_y = np.max(left_eye_region[:, 1])

    gray_eye = eye[min_y: max_y, min_x: max_x]
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    #threshold_eye = cv2.resize(threshold_eye, None, fx=5, fy=5)
    height1, width1 = threshold_eye.shape 
    
    left_side_thresh = threshold_eye[0:height1, 0:int(width1/2)]
    left_side_white = cv2.countNonZero(left_side_thresh)

    right_side_thresh = threshold_eye[0:height1, int(width1/2): width1]
    right_side_white = cv2.countNonZero(right_side_thresh)
    gaze_ratio = 0
    if left_side_white == 0:
        gaze_ratio == 1
    elif right_side_white == 0:
        gaze_ratio == 5
    else: 
        gaze_ratio = left_side_white/right_side_white

    return gaze_ratio

while True:
    _, frame = cap.read()
    new_frame = np.zeros((500, 500, 3), np.uint8)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        x, y = face.left(), face.top()
        x1, y1, = face.right(), face.bottom()
        cv2.rectangle(frame, (x, y), (x1, y1), (0, 0, 255), 2)
        landmarks = predictor(gray, face)
        left_point = (landmarks.part(36).x, landmarks.part(36).y)
        right_point = (landmarks.part(39).x, landmarks.part(39).y)
        top_center = midpoint(landmarks.part(37), landmarks.part(38))
        bottom_center = midpoint(landmarks.part(41), landmarks.part(40))

        #Gaze detection
        gaze_left_eye = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks)
        gaze_right_eye = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks)
        gaze_ratio_final = gaze_left_eye + gaze_right_eye/2
        

        if gaze_ratio_final <= 1:
            cv2.putText(frame, "RIGHT", (50, 100), font, 2, (0, 0, 255), 3)
            new_frame[:] = (0, 0, 255)
        elif 1 < gaze_ratio_final < 3:
            cv2.putText(frame, "CENTER", (50, 100), font, 2, (0, 0, 255), 3)
        else:
            cv2.putText(frame, "LEFT", (50, 100), font, 2, (0, 0, 255), 3)
            new_frame[:] = [255, 0, 0]
        # showing direction
        



        #eye = cv2.resize(eye, None, fx=5, fy=5)
        #cv2.imshow("Eye", eye)
        #cv2.imshow("Threshold", threshold_eye)
        #cv2.imshow("Left eye", left_side_thresh)
        #cv2.imshow("Right Eye", right_side_thresh)
        #x_axis = cv2.line(frame, left_point, right_point, (0, 0, 255), 2)
        #y_axis = cv2.line(frame, top_center, bottom_center, (0, 0, 255), 2)

    cv2.imshow('frame', frame)
    cv2.imshow('direction', new_frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()