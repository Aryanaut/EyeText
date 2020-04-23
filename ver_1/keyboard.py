import cv2
import numpy as np 

keyboard = np.zeros((600, 1000, 3), np.uint8)

keys_set_1 = {0:"Q",1:"W",2:"E",3:"R",4:"T",
                5:"A",6:"S",7:"D",8:"F",9:"G",
                10:"Z",11:"X",12:"C",13:"V",14:"B"}

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
    thick = 3
    size = 10
    if letter_light == True:
        cv2.rectangle(keyboard, (x+thick, y+thick), (x+width-thick, y+height-thick), (255, 255, 255), thickness=-1)
    else:
        cv2.rectangle(keyboard, (x+thick, y+thick), (x+width-thick, y+height-thick), (255, 0, 0), thickness=thick)

    #text settings
    font_scale = 10
    font_th = 4
    text_size = cv2.getTextSize(text, font, font_scale, font_th)[0]
    width_text, height_text = text_size[0], text_size[1] 
    text_x = int((width-width_text) / 2) + x
    text_y = int((height+height_text) / 2) + y

    #text
    cv2.putText(keyboard, text, (text_x, text_y), font, font_scale, (255, 0, 0), font_th)
#print(text_size)
for i in range(15):
    if i == 5:
        light = True
    else:
        light = False
    letter(i, keys_set_1[i], light)

#letter(0, "Q")
#letter(1, "W")
#letter(2, "E")
#letter(3, "R")
#letter(4, "T")
#letter(5, "A")

cv2.imshow('keyboard', keyboard)
cv2.waitKey(0)
cv2.destroyAllWindows()