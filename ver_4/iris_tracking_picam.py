import cv2
import numpy as np
import sys
import dlib
import time
import threading

GSTREAMER_PIPELINE0 = 'nvarguscamerasrc sensor_id=0 ! video/x-raw(memory:NVMM), width=3280, height=2464, format=(string)NV12, framerate=21/1 ! nvvidconv flip-method=0 ! video/x-raw, width=360, height=231, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink'
cap = cv2.VideoCapture(GSTREAMER_PIPELINE0, cv2.CAP_GSTREAMER)  
threshold = 50
def nothing(x):
    global threshold
    threshold = cv2.getTrackbarPos('threshold', 'thr')


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
frame = np.zeros((100, 100, 3), np.uint8)
gray = np.zeros((100, 100, 3), np.uint8)
cv2.createTrackbar('threshold', 'thr', 0, 255, nothing)

def midpoint(p1, p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y))

def main():
    # £capture = threading.Thread(target=get_vid)
    # capture.start()
    # global detector, predictor, frame, gray
    global threshold
    while True:
        ret, frame = cap.read()
        if ret:
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
                    min_x = np.min(left_eye_region[:, 0])
                    max_x = np.max(left_eye_region[:, 0])
                    min_y = np.min(left_eye_region[:, 1])
                    max_y = np.max(left_eye_region[:, 1])
                    left_eye = frame[min_y-5: max_y+5, min_x-3 : max_x+1]
                    left_eye = cv2.resize(left_eye, None, fx=9, fy=9)
                    gray_eye = cv2.cvtColor(left_eye, cv2.COLOR_BGR2GRAY)
                    _, thr = cv2.threshold(gray_eye, threshold, 255, cv2.THRESH_BINARY_INV)
                    
                    #thr = cv2.erode(thr, None, iterations=20)
                    #thr = cv2.dilate(thr, None, iterations=20)
                    thr = cv2.medianBlur(thr, 9)
                    thr = cv2.GaussianBlur(thr, (9, 9), 0)
                    edged = cv2.Canny(thr, 100, 200)

                    cv2.imshow('eye', edged)
                    cv2.imshow('thr', thr)                 


        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break


if __name__ == '__main__':
    main()
    cap.release()
    cv2.destroyAllWindows()