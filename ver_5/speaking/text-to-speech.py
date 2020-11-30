import pyttsx3
import cv2
import numpy as np
import time

engine = pyttsx3.init()
screen = np.zeros((200, 200, 3), np.uint8)

reg1Text = 'Hello There'
reg2Text = 'Gentlemen, off we go'
click = False
coords = (0, 0)
def onClick(event, x, y, flags, params):
    global click, coords
    if event == cv2.EVENT_LBUTTONDOWN:
        click = True
        coords = (x, y)

while True:
    cv2.rectangle(screen, (0, 0), (100, 200), (0, 0, 255), 2)
    cv2.rectangle(screen, (100, 200), (200, 200), (0, 0, 255), 2)
    cv2.putText(screen, reg1Text, (10, 10), cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 255), 1)
    cv2.putText(screen, reg2Text, (100, 10), cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 255), 1)
    if click == True:
        if coords[0] <= 100:
            engine.say(reg1Text)
            print(coords[0])
            engine.runAndWait()
            click = False
        else:
            engine.say(reg2Text)
            print(coords[0])
            engine.runAndWait()
            click = False

    cv2.setMouseCallback('screen', onClick)
    cv2.imshow('screen', screen)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
