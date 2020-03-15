import numpy as np 
import cv2

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')

#img = cv2.imread('face.jpeg')
img = cv2.VideoCapture(0)
while True:
    ret, frame = img.read()
    gray_picture = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#make picture gray
    faces = face_cascade.detectMultiScale(gray_picture, scaleFactor=1.1, minNeighbors=5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame ,(x,y),(x+w,y+h),(255,255,0),2)
        gray = gray_picture[y:y+h, x:x+w]
        face = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    height = np.size(face, 0)   
    for (ex,ey,ew,eh) in eyes:    
        if ey+eh > height/2:
            pass
        else:
            cv2.rectangle(face, (ex,ey), (ex+ew,ey+eh), (0, 0, 255), 2)
        
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
#def eye_detect(img, classifier):
    
img.release()
cv2.destroyAllWindows()