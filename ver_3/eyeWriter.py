from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QWidget, QMainWindow, QHBoxLayout, QGridLayout, QApplication, QLabel, QVBoxLayout, QBoxLayout, QPushButton, QSlider, QMessageBox
from PyQt5.QtGui import QPixmap, QImage, QColor
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QObject, QThread
import sys
import cv2
import numpy as np
import os
import pyautogui
import dlib

os.chdir('/home/aryan/Documents/Python/EyeText/ver_3')

class videoThread(QThread):
    change_pixmap_signal1=pyqtSignal(np.ndarray)
    change_pixmap_signal2=pyqtSignal(np.ndarray)
    def midpoint(self, p1 ,p2):
        return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

    def run(self):
        cap = cv2.VideoCapture(0)
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

        eyeFrame = np.zeros((150, 300, 3), np.uint8)
        while True:
            ret, img = cap.read()
            img = cv2.flip(img, 1)
            img = cv2.resize(img, None, fx=0.5, fy=0.5)
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
                    eye = cv2.resize(eye, None, fx=4, fy=4)
                    cv2.polylines(img, [left_eye_region], True, (255, 0, 255), 2)

                    ey, ex, ch = eye.shape
                    eyeFrame[int(75-(ey/2)):int(75+(ey/2)), int(150-(ex/2)):int(150+(ex/2))] = eye

            if ret:
                self.change_pixmap_signal1.emit(img)
                self.change_pixmap_signal2.emit(eyeFrame)
start = False
class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('demo')
        self.width = 640
        self.height = 480 
        self.setGeometry(10, 10, 1920, 1080)
        self.image_label = QLabel(self)
        self.eye_label = QLabel(self)
        self.textLabel = QLabel('Demo')
        self.textLabel.hide()

        self.blank = cv2.imread('stoppedvideo.jpg')

        self.button1 = QPushButton(self)
        self.button1.setText('Start Video')
        self.button1.clicked.connect(self.ifStartClicked)

        self.button2 = QPushButton(self)
        self.button2.setGeometry(0, 670, 40, 20)
        self.button2.setText('Stop Video')
        self.button2.clicked.connect(self.ifStoppedClicked)

        self.vbox = QGridLayout()
        self.vbox.addWidget(self.image_label)
        self.vbox.addWidget(self.eye_label)
        self.vbox.addWidget(self.button1)
        self.vbox.addWidget(self.button2)

        self.setLayout(self.vbox)
        self.thread = videoThread()
        self.thread.change_pixmap_signal1.connect(self.updateImage1)
        self.thread.change_pixmap_signal2.connect(self.updateImage2)
        self.thread.start()


    @pyqtSlot(np.ndarray)
    def updateImage1(self, img):
        self.qtImg = self.convertCvQt(img)
        self.image_label.setPixmap(self.qtImg)

    @pyqtSlot(np.ndarray)
    def updateImage2(self, img):
        self.qtEye = self.convertCvQt(img)
        self.eye_label.setPixmap(self.qtEye)

    def convertCvQt(self, img):
        rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgbImg.shape
        bytesPerLine = ch * w
        convertToQtFormat = QtGui.QImage(rgbImg.data, w, h, bytesPerLine, QtGui.QImage.Format_RGB888)
        #p = convertToQtFormat.scaled(self.width, self.height, Qt.KeepAspectRatio)
        p = convertToQtFormat
        return QPixmap.fromImage(p)

    def ifStartClicked(self):
        # self.textLabel.show()
        self.image_label.show()

    def ifStoppedClicked(self):
        self.image_label.hide()
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec_())