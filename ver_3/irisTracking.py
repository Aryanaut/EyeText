from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QWidget, QMainWindow, QApplication, QLabel, QVBoxLayout, QBoxLayout, QPushButton, QSlider
from PyQt5.QtGui import QPixmap, QImage, QColor
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QObject, QThread
import sys
import cv2
import numpy as np
import os
import pyautogui
import dlib

toggle = 0
threshold = 0
screenToggle = 0
click = False
point = 1

# callibration points
points = {
    1 : (0, 0),
    2 : (0, 0),
    3 : (0, 0),
    4 : (0, 0),
}

class videoThread(QThread):
    change_pixmap_signal=pyqtSignal(np.ndarray)

    def midpoint(self, p1 ,p2):
        return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

    def on_threshold(self, x):
        pass
    
    def click_pos(self, event, x, y, flags, params):
        global click
        if event == cv2.EVENT_LBUTTONDOWN:
            click = True
        else:
            click = False

    def blob_process(self, img, detection):
        img = cv2.erode(img, None, iterations=2)
        img = cv2.dilate(img, None, iterations=30)
        img = cv2.medianBlur(img, 5)
        keypoints = detection.detect(img)
        return keypoints

    def run(self):
        cap = cv2.VideoCapture(0)
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        eyeFrame = np.zeros((150, 300, 3), np.uint8)
    
        # screens
        cv2.namedWindow('eyeWindow')
        cv2.createTrackbar('threshold', 'eyeWindow', 0, 255, self.on_threshold)
        callibrationScreen = np.zeros((1080, 1920, 3), np.uint8)

        detector_params = cv2.SimpleBlobDetector_Params()
        detector_params.filterByColor = True
        detector_params.blobColor = 255
        #detector_params.filterByArea = True
        #detector_params.maxArea = 3000
        blob_detector = cv2.SimpleBlobDetector_create(detector_params)
        
        while True:
            ret, img = cap.read()
            img = cv2.flip(img, 1)
            # img = cv2.resize(img, None, fx=0.5, fy=0.5)
            grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector(grayImg)
            if len(faces) != 0:
                for face in faces:
                    # gets coordinates of the face
                    facex, facey = face.left(), face.top()
                    facex1, facey1 = face.right(), face.bottom()

                    # draws a rectangle around the face
                    cv2.rectangle(img, (facex, facey), (facex1, facey1), (0, 0, 255), 2)
                    landmarks = predictor(grayImg, face)

                    left_point = (landmarks.part(36).x, landmarks.part(36).y)
                    right_point = (landmarks.part(39).x, landmarks.part(39).y)
                    top_center = self.midpoint(landmarks.part(37), landmarks.part(38))
                    bottom_center = self.midpoint(landmarks.part(41), landmarks.part(40))
                    
                    #Gaze Ratio
                    left_eye_region = np.array([(landmarks.part(36).x, landmarks.part(36).y),
                                (landmarks.part(37).x, landmarks.part(37).y),
                                (landmarks.part(38).x, landmarks.part(38).y),
                                (landmarks.part(39).x, landmarks.part(39).y),
                                (landmarks.part(40).x, landmarks.part(40).y),
                                (landmarks.part(41).x, landmarks.part(41).y)], np.int32)
                    
                    min_x = np.min(left_eye_region[:, 0])
                    max_x = np.max(left_eye_region[:, 0])
                    min_y = np.min(left_eye_region[:, 1])
                    max_y = np.max(left_eye_region[:, 1])
                    
                    eye = img[min_y-1: max_y, min_x : max_x]
                    eye = cv2.resize(eye, None, fx=3, fy=3)
                    gray_eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
                    threshold = cv2.getTrackbarPos('threshold', 'eyeWindow')
                    _, eyeImg = cv2.threshold(gray_eye, threshold, 255, cv2.THRESH_BINARY_INV)
                    keypoints = self.blob_process(eyeImg, blob_detector)
                    # print(keypoints)
                    for keypoint in keypoints:
                        s = keypoint.size
                        x = keypoint.pt[0]
                        y = keypoint.pt[1]
                        cx = int(x)
                        cy = int(y)
                        coordinates = (cx, cy)
                        cv2.circle(eye, (cx, cy), 5, (0, 0, 255), -1)

                    cv2.drawKeypoints(eye, keypoints, eye, (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                    cv2.polylines(img, [left_eye_region], True, (255, 0, 255), 2)
                
                # print(eye.shape)
                ey, ex, ch = eye.shape
                eyeFrame[0:ey, 0:ex] = eye
                # cv2.imshow('eyeWindow', eye)
                if toggle%2 == 1:
                    cv2.imshow('eyeWindow', eyeFrame)
                if toggle%2 == 0 & toggle !=0:
                    cv2.destroyWindow('eyeWindow')

                if screenToggle%2 == 1:
                    global point
                    cv2.putText(callibrationScreen, "UP", (900, 60), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 2)
                    cv2.putText(callibrationScreen, "RIGHT", (1700, 540), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 2)
                    cv2.putText(callibrationScreen, "LEFT", (20, 540), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 2)
                    cv2.putText(callibrationScreen, "DOWN", (860, 1000), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 2)
                    
                    if click == True:
                        points[point] = coordinates
                        point = point+1
                        print(points)

                    cv2.setMouseCallback('Callibration Screen', self.click_pos)
                    cv2.imshow('Callibration Screen', callibrationScreen)
                if screenToggle%2 == 0 & screenToggle !=0:
                    cv2.destroyWindow('Callibration Screen')

            if ret:
                self.change_pixmap_signal.emit(img)
                # self.change_pixmap_signal.emit(eye)

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Main Window')
        self.width = 640
        self.height = 480
        self.setGeometry(10, 10, 1920, 1080)
        self.frameLabel = QLabel(self)
        '''
        self.left = QLabel('LEFT')
        self.left.setGeometry(960, 20, 30, 15)
        self.right = QLabel('RIGHT')
        self.up = QLabel('UP')
        self.down = QLabel('DOWN')
        '''
        self.setStyleSheet('background-color: rgb(40, 40, 40); color: white')
        self.frameLabel.hide()
        self.uiStyle = 'background-color: rgb(70, 70, 70); border-radius:10px; border-color: beige; padding: 6px'

        # buttons
        self.startButton = QPushButton(self)
        self.startButton.setText('Start Video')
        self.startButton.clicked.connect(self.ifStartVideoClicked)
        self.startButton.setStyleSheet(self.uiStyle)
        self.startButton.hide()

        self.startProgram = QPushButton(self)
        self.startProgram.setText('Start Program')
        self.startProgram.clicked.connect(self.ifStartProgramClicked)
        self.startProgram.setStyleSheet(self.uiStyle)

        self.toggleEye = QPushButton(self)
        self.toggleEye.setText('Toggle Eye Win')
        self.toggleEye.clicked.connect(self.toggleEyeWin)
        self.toggleEye.hide()
        self.toggleEye.setStyleSheet(self.uiStyle)

        self.toggleCallibration = QPushButton(self)
        self.toggleCallibration.setText('Callibrate')
        self.toggleCallibration.clicked.connect(self.ifToggleCallibrationClicked)
        self.toggleCallibration.hide()
        self.toggleCallibration.setStyleSheet(self.uiStyle)

        # layout (add buttons here)
        self.vbox = QVBoxLayout()
        self.vbox.addWidget(self.frameLabel)
        self.vbox.addWidget(self.startButton)
        self.vbox.addWidget(self.startProgram)
        self.vbox.addWidget(self.toggleEye)
        self.setLayout(self.vbox)
        # self.setCentralWidget(self.toggleEye)

    @pyqtSlot(np.ndarray)
    def updateImage(self, img):
        self.qtImg = self.convertCvQt(img)
        self.frameLabel.setPixmap(self.qtImg)

    def convertCvQt(self, img):
        rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgbImg.shape
        bytesPerLine = ch * w
        convertToQtFormat = QtGui.QImage(rgbImg.data, w, h, bytesPerLine, QtGui.QImage.Format_RGB888)
        p = convertToQtFormat.scaled(self.width, self.height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def ifStartVideoClicked(self):
        self.frameLabel.show()
        self.toggleEye.show()
        self.toggleCallibration.show()
        self.startProgram.hide()
        self.startButton.hide()

    def ifStartProgramClicked(self):
        # grabs the frames from the camera
        self.thread = videoThread()
        self.thread.change_pixmap_signal.connect(self.updateImage)
        self.thread.start()
        self.startButton.show()

    def toggleEyeWin(self):
        global toggle
        toggle = toggle + 1
        # print(toggle)
    
    def ifToggleCallibrationClicked(self):
        global screenToggle
        screenToggle += 1
    
        
if __name__== '__main__':
    app = QApplication(sys.argv)
    a = App()
    app.setStyle('Breeze')
    a.show()
    sys.exit(app.exec_())
    cv2.destroyAllWindows()