import cv2
import numpy as np


def detectClick(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN: 
        click = True
        print('ok')


click = False
screen = np.zeros((600, 800, 3), np.uint8)


while True:
    # print(click)
    cv2.imshow('screen', screen)
    cv2.setMouseCallback('screen', detectClick)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cv2.destroyAllWindows()
