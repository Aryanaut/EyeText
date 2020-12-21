import cv2
import numpy as np
import dlib

# definitions
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
frame = np.zeros((100, 100, 3), np.uint8)
gray = np.zeros((100, 100, 3), np.uint8)
cv2.createTrackbar('threshold', 'thr', 0, 255, nothing)
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
            gray = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            faces = detector(gray)
            if str(faces) == 'rectangles':
                pass
            else:
                for face in faces:
                    x, y = face.left(), face.top()
                    x1, y1 = face.right(), face.bottom()
                    cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

                    landmarks = predictor(gray, face)
            cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
        


if __name__ == '__main__':
    main()

cv2.destroyAllWindows()