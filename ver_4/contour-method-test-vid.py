import cv2
import numpy as np
import dlib

cap = cv2.VideoCapture('eye-movement.mp4')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def midpoint(p1, p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y))

while True:
    ret, frame = cap.read()
    if ret is False: 
        break
    frame = cv2.resize(frame, None, fx=0.3, fy=0.3)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if str(faces) == 'rectangles[]':
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

            right_eye_region = np.array([(landmarks.part(42).x, landmarks.part(42).y),
                                (landmarks.part(43).x, landmarks.part(43).y),
                                (landmarks.part(44).x, landmarks.part(44).y),
                                (landmarks.part(45).x, landmarks.part(45).y),
                                (landmarks.part(46).x, landmarks.part(46).y),
                                (landmarks.part(47).x, landmarks.part(47).y)], np.int32)

            min_x = np.min(left_eye_region[:, 0])
            max_x = np.max(left_eye_region[:, 0])
            min_y = np.min(left_eye_region[:, 1])
            max_y = np.max(left_eye_region[:, 1])

            min_x1 = np.min(right_eye_region[:, 0])
            max_x1 = np.max(right_eye_region[:, 0])
            min_y1 = np.min(right_eye_region[:, 1])
            max_y1 = np.max(right_eye_region[:, 1])

            left_eye = frame[min_y-1: max_y+3, min_x-3 : max_x+1]
            right_eye = frame[min_y1-1: max_y1+3, min_x1-3 : max_x1+1]
            right_eye = cv2.resize(right_eye, None, fx=8, fy=8)
            left_eye = cv2.resize(left_eye, None, fx=8, fy=8)
            gray_eye = cv2.cvtColor(right_eye, cv2.COLOR_BGR2GRAY)
            _, thr = cv2.threshold(gray_eye, 80, 255, cv2.THRESH_BINARY_INV)
            
            edged = cv2.Canny(thr, 80, 200)
            thr = cv2.GaussianBlur(thr, (3, 3), 0)
            thr = cv2.dilate(thr, None, iterations=6)
            thr = cv2.erode(thr, None, iterations=2)
            thr = cv2.GaussianBlur(thr, (9, 9), 0)
            
            cont, h = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(right_eye, cont, -1, (0, 255, 0), 3)
            largest = max(cont, key=cv2.contourArea)
            hull = []
            
            for i in range(len(cont)):
                hull.append(cv2.convexHull(largest, False))
                cv2.drawContours(right_eye, hull, i, (0, 0, 255), 3)

            cv2.polylines(frame, [left_eye_region], True, (0, 0, 255), 3)
            cv2.polylines(frame, [right_eye_region], True, (0, 0, 255), 3)


    cv2.imshow('right_eye_region', right_eye)
    cv2.imshow('thr', thr)
    cv2.imshow('edged', edged)
    if cv2.waitKey(40) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()