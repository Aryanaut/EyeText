import cv2
import numpy as np
import dlib

def nothing(x):
    pass

# definitions
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
frame = np.zeros((100, 100, 3), np.uint8)
gray = np.zeros((100, 100, 3), np.uint8)
cv2.namedWindow('frame')
cv2.createTrackbar('threshold', 'frame', 55, 255, nothing)
cv2.createTrackbar('dilate', 'frame', 15, 30, nothing)
cv2.createTrackbar('erode', 'frame', 15, 30, nothing)
board = np.zeros((297, 420, 3), np.uint8)
board = cv2.resize(board, None, fx=3, fy=3)

GSTREAMER_PIPELINE0 = 'nvarguscamerasrc sensor_id=0 ! video/x-raw(memory:NVMM), width=3280, height=2464, format=(string)NV12, framerate=21/1 ! nvvidconv flip-method=0 ! video/x-raw, width=360, height=231, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink'
cap = cv2.VideoCapture(GSTREAMER_PIPELINE0, cv2.CAP_GSTREAMER) 

def midpoint(p1, p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y))

def main():
    while True:
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            if str(faces) == 'rectangles':
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
                    left_eye = frame[min_y-1: max_y+3, min_x-3 : max_x+1] 
                    left_eye = cv2.resize(left_eye, None, fx=8, fy=8)
                    thresh = cv2.getTrackbarPos('threshold', 'frame')
                    erosion = cv2.getTrackbarPos('erode', 'frame')
                    dilation = cv2.getTrackbarPos('dilate', 'frame')
                    gray_eye = cv2.cvtColor(left_eye, cv2.COLOR_BGR2GRAY)
                    _, thr = cv2.threshold(gray_eye, thresh, 255, cv2.THRESH_BINARY_INV) 
                    thr = cv2.erode(thr, None, iterations=erosion)
                    thr = cv2.dilate(thr, None, iterations=dilation)
                    thr = cv2.medianBlur(thr, 9)
                    thr = cv2.GaussianBlur(thr, (9, 9), 0)
                    cv2.imshow('eye', thr)

            cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
        


if __name__ == '__main__':
    main()

cv2.destroyAllWindows()
