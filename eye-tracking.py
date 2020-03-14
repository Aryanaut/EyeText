import cv2
import numpy as np 

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye_tree_eyeglasses.xml')
cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
        print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w]
        color = (255, 0, 0)
        stroke = 2
        width = x+w
        height = y+h    
        cv2.rectangle(frame, (x,y), (width, height), color, thickness=stroke)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
