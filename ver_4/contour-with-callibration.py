import cv2
import numpy as np 
import argparse
import dlib
import threading

ap = argparse.ArgumentParser()
ap.add_argument('-t', '--streamType', type=str, required=True,
                help='Choose between -picam- and -webcam-')
args = vars(ap.parse_args())

streamType = args['streamType']
if streamType == 'picam':
    GSTREAMER_PIPELINE0 = 'nvarguscamerasrc sensor_id=0 ! video/x-raw(memory:NVMM), width=3280, height=2464, format=(string)NV12, framerate=21/1 ! nvvidconv flip-method=0 ! video/x-raw, width=960, height=616, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink'
    cap = cv2.VideoCapture(GSTREAMER_PIPELINE0, cv2.CAP_GSTREAMER)
else:
    cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
cv2.namedWindow('frame')
frame = np.zeros((1000, 1000, 3), np.uint8)
    
def capture():
    global frame
    while True:
        _, frame = cap.read()
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5)

def midpoint(p1, p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y))

def main():
    f = threading.Thread(target=capture)
    f.start()
    global cap, frame
    while True:
        try:
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

                    # left eye
                    min_x = np.min(left_eye_region[:, 0])
                    max_x = np.max(left_eye_region[:, 0])
                    min_y = np.min(left_eye_region[:, 1])
                    max_y = np.max(left_eye_region[:, 1])

                    left_eye = frame[min_y-1: max_y+3, min_x-3 : max_x+1]
                    left_eye = cv2.resize(left_eye, None, fx=4, fy=4)

                    left_gray = cv2.cvtColor(left_eye, cv2.COLOR_BGR2GRAY)
                    _, thr = cv2.threshold(left_gray, 60, 255, cv2.THRESH_BINARY)
                    edged = cv2.Canny(thr, 80, 200)
                    thr = cv2.GaussianBlur(thr, (9, 9), 2)

                    cont, h = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                    if len(cont) > 0:
                        largest = max(cont, key = cv2.contourArea)
                        cv2.drawContours(left_eye, cont, -1, (0, 0, 255), 3)

                    cv2.polylines(frame, [left_eye_region], True, (255, 255, 0), 3)
                    cv2.imshow('left_eye', left_eye)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xff == ord('q'):
                break

        except KeyboardInterrupt:
            break
        except EnvironmentError:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

 