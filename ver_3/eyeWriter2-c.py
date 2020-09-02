from PyQt5 import QtWidgets, QtGui, QtCore, uic
from PyQt5.QtWidgets import QWidget, QMainWindow, QHBoxLayout, QGridLayout, QApplication, QLabel, QVBoxLayout, QBoxLayout, QPushButton, QSlider, QMessageBox
from PyQt5.QtGui import QPixmap, QImage, QColor
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QObject, QThread, QSize
import sys
import cv2
import numpy as np
import os
import pyautogui
import dlib
import threading
import logging

os.chdir('/home/aryan/Documents/Python/EyeText/ver_3')
eyeCoordinatesAtCenter = (0, 0)
toggleTracking = False
toggleMenu = True
screenX, screenY = pyautogui.size()

class videoThread(QThread):
    global screenX, screenY
    change_frame_pixmap_signal=pyqtSignal(np.ndarray)
    change_eye_pixmap_signal=pyqtSignal(np.ndarray)
    change_tracking_pixmap_signal=pyqtSignal(np.ndarray)

    callibrationScreen = np.zeros((screenY, screenX, 3), np.uint8)
    threshold = 0
    zoom = 5
    coordinates = (0, 0)
    listOfCoords = []
    callibrationStatus = False
    lines = False
    indexCoords = 0

    centerCoords = (int(screenX/2), int(screenY/2))
    testTracking = np.zeros((screenY, screenX, 3), np.uint8)
    testTracking[:] = 0
    reg1 = [(0, 0), (683, 384)]
    reg2 = [(683, 0), (1366, 384)]
    reg3 = [(683, 384), (1366, 768)]
    reg4 = [(0, 384), (683, 768)]

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

    def regionSelector(self, gazeX, gazeY):
        if gazeX > self.centerCoords[0]:
            if gazeY > self.centerCoords[1]:
                return self.reg3
            if gazeY < self.centerCoords[1]:
                return self.reg2
        if gazeX < self.centerCoords[0]:
            if gazeY > self.centerCoords[1]:
                return self.reg4
            if gazeY < self.centerCoords[1]:
                return self.reg1

    def run(self):
        cap = cv2.VideoCapture(0)
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

        eyeFrame = np.zeros((150, 300, 3), np.uint8)
        eyeFrame[:] = 40
        screenX, screenY = pyautogui.size()
        detector_params = cv2.SimpleBlobDetector_Params()
        gazeX, gazeY = (0, 0)

        detector_params.filterByColor = True
        detector_params.blobColor = 255
        #detector_params.filterByArea = True
        #detector_params.maxArea = 3000
        blob_detector = cv2.SimpleBlobDetector_create(detector_params)
        global eyeCoordinatesAtCenter
        listofGazeCoords = [(int(screenX/2), int(screenY/2)), (int(screenX/2), int(screenY/2))]
        gazeCoordsIndex = 1
        frame = 0

        while True:
            ret, img = cap.read()
            frame += 1
            img = cv2.flip(img, 1)
            img = cv2.resize(img, None, fx=0.5, fy=0.5)
            grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector(grayImg)
            if len(faces) != 0:
                for face in faces:
                    facex, facey = face.left(), face.top()
                    facex1, facey1 = face.right(), face.bottom()

                    # draws a rectangle around the face
                    cv2.rectangle(img, (facex, facey), (facex1, facey1),
                        (0, 0, 255), thickness=2)
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
                    eye = cv2.resize(eye, None, fx=self.zoom, fy=self.zoom)
                    
                    gray_eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
                    gray_eye = cv2.GaussianBlur(gray_eye, (7, 7), 0)
                    threshold = cv2.getTrackbarPos('threshold', 'eyeWindow')
                    _, eyeImg = cv2.threshold(gray_eye, self.threshold, 255, cv2.THRESH_BINARY_INV)
                    keypoints = self.blob_process(eyeImg, blob_detector)
                    # print(keypoints)
                    for keypoint in keypoints:
                        s = keypoint.size
                        x = keypoint.pt[0]
                        y = keypoint.pt[1]
                        cx = int(x)
                        cy = int(y)
                        self.coordinates = (cx, cy)
                        cv2.circle(eye, (cx, cy), 5, (0, 0, 255), -1)
                        self.listOfCoords.append(self.coordinates)
                    
                    cv2.drawKeypoints(eye, keypoints, eye, (0, 255, 0), 
                            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

                    cv2.polylines(img, [left_eye_region], True, (255, 0, 255), 2)
                    ey, ex, ch = eye.shape
                    eyeFrame[int(75-(ey/2)):int(75+(ey/2)), int(150-(ex/2)):int(150+(ex/2))] = eye

                    # calculating coordinates for the point on the screen
                    # equation - (x/w)*w1, (y/h)*h1
                    '''
                    gazeX = int((self.coordinates[0]/ex)*w1)
                    gazeY = int((self.coordinates[1]/ey)*h1)
                    if gazeY > centerCoords[1]:
                        gazeY = gazeY-100
                    if gazeY < centerCoords[1]:
                        gazeY = gazeY+100
                    if gazeY == 0:
                        gazeY = 0

                    if gazeX > centerCoords[0]:
                        gazeX = gazeX+50
                    if gazeX < centerCoords[0]:
                        gazeX = gazeX-50

                    gazePosition = (gazeX, gazeY)
                    listofGazeCoords.append(gazePosition)
                    # cv2.circle(testTracking, gazePosition, 3, (0, 255, 0), -1)
                    if self.lines == True:
                        cv2.line(self.testTracking, listofGazeCoords[gazeCoordsIndex-1], listofGazeCoords[gazeCoordsIndex], (0, 255, 0), 2)
                    if self.lines == False:
                        self.testTracking[:] = 0
                    gazeCoordsIndex += 1
                    '''
                    # print(eyeCoordinatesAtCenter, self.coordinates)
                    if self.callibrationStatus == True:
                        # self.testTracking()
                        xDiff = self.coordinates[0]-eyeCoordinatesAtCenter[0]
                        yDiff = self.coordinates[1]-eyeCoordinatesAtCenter[1]
                        xRatio = int(self.centerCoords[0]/eyeCoordinatesAtCenter[0])
                        yRatio = int(self.centerCoords[1]/eyeCoordinatesAtCenter[1])
                        if xDiff > 0:
                            gazeX = self.centerCoords[0]+(xDiff*xRatio)
                        if xDiff < 0:
                            gazeX = self.centerCoords[0]+(xDiff*xRatio)
                        if yDiff > 0:
                            gazeY = self.centerCoords[1]-(yDiff*yRatio*4)
                        if yDiff < 0:
                            gazeY = self.centerCoords[1]+(yDiff*yRatio*4)
                        gazePosition = (gazeX, gazeY)
                        listofGazeCoords.append(gazePosition)
                        activeRegion = self.regionSelector(gazeX, gazeY)
                        if self.lines == True:
                            cv2.line(self.testTracking, listofGazeCoords[gazeCoordsIndex-1], listofGazeCoords[gazeCoordsIndex], (0, 255, 0), 2)
                        if self.lines == False:
                            self.testTracking[:] = 0

                        if activeRegion == self.reg1:
                            cv2.rectangle(self.testTracking, activeRegion[0], activeRegion[1], (255, 255, 255), 3)
                        if activeRegion == self.reg2:
                            cv2.rectangle(self.testTracking, activeRegion[0], activeRegion[1], (255, 255, 255), 3)
                        if activeRegion == self.reg3:
                            cv2.rectangle(self.testTracking, activeRegion[0], activeRegion[1], (255, 255, 255), 3)
                        if activeRegion == self.reg4:
                            cv2.rectangle(self.testTracking, activeRegion[0], activeRegion[1], (255, 255, 255), 3)

                        gazeCoordsIndex += 1
            if ret:
                self.change_frame_pixmap_signal.emit(img)
                self.change_eye_pixmap_signal.emit(eyeFrame)
                self.change_tracking_pixmap_signal.emit(self.testTracking)


class App(QtWidgets.QMainWindow):
    def __init__(self):
        super(App, self).__init__()
        global screenX, screenY, toggleTracking
        if (screenX, screenY) == (1920, 1080):
            uic.loadUi('eyeWriterInterface-largeMonitor.ui', self)
        else:
            uic.loadUi('eyeWriterInterface-smallMonitor.ui', self)

        self.setGeometry(0, 0, screenX, screenY)
        self.xIcon = self.findChild(QtWidgets.QLabel, 'xIcon')
        self.xIcon.hide()
        self.testTracking = self.findChild(QtWidgets.QLabel, 'testTracking')
        self.testTracking.setGeometry(0, 0, screenX, screenY)
        self.testTracking.hide()

        self.buttons = self.findChild(QtWidgets.QWidget, 'buttons')
        self.sliders = self.findChild(QtWidgets.QWidget, 'sliders')
        self.buttons.show()
        self.sliders.show()

        self.frame_label = self.findChild(QtWidgets.QLabel, 'frame_label')
        self.frame_label.hide()
        self.eye_label = self.findChild(QtWidgets.QLabel, 'eye_label')
        self.eye_label.hide()

        self.thresholdSlider = self.findChild(QtWidgets.QSlider, 'thresholdSlider')
        self.thresholdSlider.valueChanged[int].connect(self.thresholdChanged)

        self.zoomSlider = self.findChild(QtWidgets.QSlider, 'zoomSlider')
        self.zoomSlider.valueChanged[int].connect(self.zoomChanged)

        self.startVideo = self.findChild(QtWidgets.QPushButton, 'startVideo')
        self.startVideo.clicked.connect(self.startVideoClicked)

        self.menuButton = self.findChild(QtWidgets.QPushButton, 'menuButton')
        self.menuButton.clicked.connect(self.menuClicked)
        self.menuButton.setIcon(QtGui.QIcon('gear-icon.png'))
        self.menuButton.setIconSize(QSize(40, 40))

        self.refreshTestScreen = self.findChild(QtWidgets.QPushButton, 'refreshTestScreen')
        self.refreshTestScreen.clicked.connect(self.refreshScreen)

        self.startTest = self.findChild(QtWidgets.QPushButton, 'startTest')
        self.startTest.clicked.connect(self.startTestClicked)

        self.toggleLines = self.findChild(QtWidgets.QPushButton, 'toggleLines')
        self.toggleLines.clicked.connect(self.toggleLinesClicked)

        self.clickCenter = self.findChild(QtWidgets.QPushButton, 'clickCenter')
        self.clickCenter.clicked.connect(self.ifCenterClicked)
        self.clickCenter.hide()

        self.callibrate = self.findChild(QtWidgets.QPushButton, 'callibrate')
        self.callibrate.clicked.connect(self.callibrateClicked)

        self.thread = videoThread()
        self.thread.change_frame_pixmap_signal.connect(self.updateFrameImage)
        self.thread.change_eye_pixmap_signal.connect(self.updateEyeImage)
        self.thread.change_tracking_pixmap_signal.connect(self.updateTrackingImage)
        self.thread.start()
        self.show()

    @pyqtSlot(np.ndarray)
    def updateFrameImage(self, img):
        self.qtImg = self.convertCvQt(img)
        self.frame_label.setPixmap(self.qtImg)

    @pyqtSlot(np.ndarray)
    def updateEyeImage(self, img):
        self.eyeImg = self.convertCvQt(img)
        self.eye_label.setPixmap(self.eyeImg)

    @pyqtSlot(np.ndarray)
    def updateTrackingImage(self, img):
        self.trackingImg = self.convertCvQt(img)
        self.testTracking.setPixmap(self.trackingImg)

    def convertCvQt(self, img):
        rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgbImg.shape
        bytesPerLine = ch * w
        convertToQtFormat = QtGui.QImage(rgbImg.data, w, h, bytesPerLine, QtGui.QImage.Format_RGB888)
        #p = convertToQtFormat.scaled(self.width, self.height, Qt.KeepAspectRatio)
        p = convertToQtFormat
        return QPixmap.fromImage(p)

    def startVideoClicked(self):
        self.frame_label.show()
        self.eye_label.show()

    def thresholdChanged(self, value):
        self.thread.threshold = value

    def zoomChanged(self, value):
        self.thread.zoom = value

    def callibrateClicked(self):
        self.xIcon.show()
        self.clickCenter.show()
        self.callibrationInstructions = QMessageBox.about(self, 'Instructions', 'Look at the X on the screen and click the CENTER button')
    
    def startTestClicked(self):
        global toggleTracking
        toggleTracking = not toggleTracking
        if toggleTracking == True:
            self.testTracking.show()
        else:
            self.testTracking.hide()
        print(toggleTracking)

    def ifCenterClicked(self):
        global eyeCoordinatesAtCenter
        if self.thread.coordinates == (0, 0):
            QMessageBox.about(self, 'Try Again')
        else:
            eyeCoordinatesAtCenter = self.thread.coordinates
            print(eyeCoordinatesAtCenter)
        # print(eyeCoordinatesAtCenter)
        self.thread.callibrationStatus = True
        self.xIcon.hide()
        self.clickCenter.hide()

    def menuClicked(self):
        global toggleMenu
        toggleMenu = not toggleMenu
        if toggleMenu == False:
            self.buttons.hide()
            self.sliders.hide()
        else:
            self.buttons.show()
            self.sliders.show()

    def refreshScreen(self):
        self.thread.testTracking[:] = 0

    def toggleLinesClicked(self):
        self.thread.lines = not self.thread.lines


    

if __name__ == "__main__":
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec_())