import cv2
import numpy as np
import dlib
from math import hypot
from playsound import playsound

gaze_counter = 0

cap = cv2.VideoCapture(0)
board = np.zeros((500, 500), np.uint8)
board[:] = 255
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

menu = np.zeros((600, 600), np.uint8)


#keyboard settings
keyboard = np.zeros((600, 1000, 3), np.uint8)
keyboard1 = np.zeros((600, 1000, 3), np.uint8)

keys_set_1 = {0:"Q",1:"W",2:"E",3:"R",4:"T",
                5:"A",6:"S",7:"D",8:"F",9:"G",
                10:"Z",11:"X",12:"C",13:"V",14:"B"}

keys_set_2 = {0:"Y",1:"U",2:"I",3:"O",4:"P",
                5:"H",6:"J",7:"K",8:"L",9:":",
                10:"N",11:"M",12:" ",13:"."}

def letter(letter_index, text, letter_light):
    #keys
    if letter_index == 0:
        x = 0
        y = 0
    elif letter_index == 1:
        x = 200
        y = 0
    elif letter_index == 2:
        x = 400
        y = 0
    elif letter_index ==3:
        x = 600
        y = 0
    elif letter_index == 4:
        x = 800
        y = 0
    elif letter_index == 5:
        x = 0
        y = 200
    elif letter_index == 6:
        x = 200
        y = 200
    elif letter_index == 7:
        x = 400
        y = 200
    elif letter_index == 8:
        x = 600
        y = 200        
    elif letter_index == 9:
        x = 800
        y = 200
    elif letter_index == 10:
        x = 0
        y = 400
    elif letter_index == 11:
        x = 200
        y = 400
    elif letter_index == 12:
        x = 400
        y = 400
    elif letter_index == 13:
        x = 600
        y = 400
    elif letter_index == 14:
        x = 800
        y = 400
    font = cv2.FONT_HERSHEY_PLAIN
    width = 200
    height = 200
    thick = 2
    size = 10
    if letter_light == True:
        cv2.rectangle(keyboard, (x+thick, y+thick), (x+width-thick, y+height-thick), (150, 150, 150), -1)
    else:
        cv2.rectangle(keyboard, (x+thick, y+thick), (x+width-thick, y+height-thick), (255, 255, 255), thickness=thick)

    #text settings
    font_scale = 10
    font_th = 4
    text_size = cv2.getTextSize(text, font, font_scale, font_th)[0]
    width_text, height_text = text_size[0], text_size[1] 
    text_x = int((width-width_text) / 2) + x
    text_y = int((height+height_text) / 2) + y

    #text
    cv2.putText(keyboard, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_th)

def letterRight(letter_index, text, letter_light):
    if letter_index == 0:
        x = 0
        y = 0
    elif letter_index == 1:
        x = 200
        y = 0
    elif letter_index == 2:
        x = 400
        y = 0
    elif letter_index ==3:
        x = 600
        y = 0
    elif letter_index == 4:
        x = 800
        y = 0
    elif letter_index == 5:
        x = 0
        y = 200
    elif letter_index == 6:
        x = 200
        y = 200
    elif letter_index == 7:
        x = 400
        y = 200
    elif letter_index == 8:
        x = 600
        y = 200        
    elif letter_index == 9:
        x = 800
        y = 200
    elif letter_index == 10:
        x = 0
        y = 400
    elif letter_index == 11:
        x = 200
        y = 400
    elif letter_index == 12:
        x = 400
        y = 400
    elif letter_index == 13:
        x = 600
        y = 400
    elif letter_index == 14:
        x = 800
        y = 400
    font = cv2.FONT_HERSHEY_PLAIN
    width = 200
    height = 200
    thick = 2
    size = 10
    if letter_light == True:
        cv2.rectangle(keyboard, (x+thick, y+thick), (x+width-thick, y+height-thick), (150, 150, 150), -1)
    else:
        cv2.rectangle(keyboard, (x+thick, y+thick), (x+width-thick, y+height-thick), (255, 255, 255), thickness=thick)

    #text settings
    font_scale = 10
    font_th = 4
    text_size = cv2.getTextSize(text, font, font_scale, font_th)[0]
    width_text, height_text = text_size[0], text_size[1] 
    text_x = int((width-width_text) / 2) + x
    text_y = int((height+height_text) / 2) + y

    #text
    cv2.putText(keyboard, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_th)

def midpoint(p1, p2):
    return int((p1.x+p2.x)/2), int((p1.y+p2.y)/2)

font = cv2.FONT_HERSHEY_PLAIN

def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))
    #hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
    #ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)
    hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
    ratio = hor_line_lenght / ver_line_lenght
    return ratio

def get_gaze_ratio(eye_points, facial_landmarks):
    left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                            (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                            (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                            (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                            (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                            (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)
    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [left_eye_region], True, 255, 2)
    cv2.fillPoly(mask, [left_eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)

    min_x = np.min(left_eye_region[:, 0])
    max_x = np.max(left_eye_region[:, 0])
    min_y = np.min(left_eye_region[:, 1])
    max_y = np.max(left_eye_region[:, 1])

    gray_eye = eye[min_y: max_y, min_x: max_x]
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    #threshold_eye = cv2.resize(threshold_eye, None, fx=5, fy=5)
    height1, width1 = threshold_eye.shape 
    
    left_side_thresh = threshold_eye[0:height1, 0:int(width1/2)]
    left_side_white = cv2.countNonZero(left_side_thresh)

    right_side_thresh = threshold_eye[0:height1, int(width1/2): width1]
    right_side_white = cv2.countNonZero(right_side_thresh)
    gaze_ratio = 0
    if left_side_white == 0:
        gaze_ratio == 1
    elif right_side_white == 0:
        gaze_ratio == 5
    else: 
        gaze_ratio = left_side_white/right_side_white

    return gaze_ratio


#Counters
frames = 0
letter_index = 0
blinking_frames = 0
text = ""

while True:
    _, frame = cap.read()
    frame = cv2.resize(frame, None, fx=0.5,fy=0.5)
    keyboard[:] = (50, 50, 50)
    frames+=1
    new_frame = np.zeros((500, 500, 3), np.uint8)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    active_letter = keys_set_1[letter_index]

    faces = detector(gray)
    for face in faces:
        x, y = face.left(), face.top()
        x1, y1, = face.right(), face.bottom()
        cv2.rectangle(frame, (x, y), (x1, y1), (0, 0, 255), 2)
        landmarks = predictor(gray, face)

        left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

        if blinking_ratio > 5.7:
            cv2.putText(frame, "BLINKING", (50, 150), font, 4, (255, 0, 0), thickness=3)
            blinking_frames += 1
            frames -= 1

            if blinking_frames == 3:
                playsound('Ding-sound-effect.mp3')
                text += active_letter
        else:
            blinking_frames = 0

        left_point = (landmarks.part(36).x, landmarks.part(36).y)
        right_point = (landmarks.part(39).x, landmarks.part(39).y)
        top_center = midpoint(landmarks.part(37), landmarks.part(38))
        bottom_center = midpoint(landmarks.part(41), landmarks.part(40))

        #Gaze detection
        gaze_left_eye = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks)
        gaze_right_eye = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks)
        gaze_ratio_final = gaze_left_eye + gaze_right_eye/2
        
        
        if gaze_ratio_final <= 1:
            cv2.putText(frame, "RIGHT", (50, 100), font, 2, (0, 0, 255), 3)
            new_frame[:] = (0, 0, 255)
        elif 1 < gaze_ratio_final < 3:
            cv2.putText(frame, "CENTER", (50, 100), font, 2, (0, 0, 255), 3)
        else:
            cv2.putText(frame, "LEFT", (50, 100), font, 2, (0, 0, 255), 3)
            new_frame[:] = [255, 0, 0]
        # showing direction
        #eye = cv2.resize(eye, None, fx=5, fy=5)
        #cv2.imshow("Eye", eye)
        #cv2.imshow("Threshold", threshold_eye)
        #cv2.imshow("Left eye", left_side_thresh)
        #cv2.imshow("Right Eye", right_side_thresh)
        #x_axis = cv2.line(frame, left_point, right_point, (0, 0, 255), 2)
        #y_axis = cv2.line(frame, top_center, bottom_center, (0, 0, 255), 2)
        

    #Letters
    if frames == 20:
        #print("10", keys_set_1[letter_index])
        letter_index+=1
        frames = 0
    if letter_index == 15:
        letter_index = 0
    for i in range(15):
        if i == letter_index:
            light = True
        else:
            light = False
        letter(i, keys_set_1[i], light)

    cv2.putText(board, text, (10, 100), font, 4, (0, 0, 0), 3)

    #What is shown
    cv2.imshow('frame', frame)
    cv2.imshow('direction', new_frame)
    cv2.imshow('left-keyboard', keyboard)
    cv2.imshow('board', board)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()