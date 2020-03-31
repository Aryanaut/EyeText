import cv2
import numpy as np
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

img = cv2.imread('face.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')

#blob parameters

faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

def face_detect(image, cascade):
    gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    coords = cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
    if len(coords) > 1:
        largest = (0, 0, 0, 0)
        for i in coords:
            if i[3] > largest[3]:
                largest = i
        largest = np.array([i], np.int32)
    elif len(coords) == 1:
        largest = coords
    else:
        return None
    for (x, y, w, h) in largest:
        frame = image[y:y+h, x:x+w]
    return frame

#for(x, y, w, h) in faces:
#    cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)
#    roi_gray = gray[y:y+h, x:x+w]
#    roi = img[y:y+h, x:x+w]
#    eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.3, minNeighbors=5)

def eye_detect(image, cascade):
    gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eyes = cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
    height = np.size(image, 0)
    width = np.size(image, 1)
    leftEye = None
    rightEye = None
    for(x, y, w, h) in eyes:
        if y > height/2:
            pass
        eyecenter = x+w/2
        if eyecenter < width*0.5:
            leftEye = image[y:y+h, x:x+w]
        else:
            rightEye = image[y:y+h, x:x+w]
    return leftEye, rightEye

face_detect(img, face_cascade)
eye_detect(img, eye_cascade)

cv2.imshow('my image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()