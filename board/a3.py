import cv2
import numpy as np

board = np.zeros((297, 420, 3), np.uint8)
board = cv2.resize(board, None, fx=3, fy=3)
while True:
    cv2.imshow('board', board)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
        
cv2.destroyAllWindows()