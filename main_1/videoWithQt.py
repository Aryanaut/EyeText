from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QHBoxLayout, QPushButton
from PyQt5.QtGui import QPixmap, QImage, QColor
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QObject, QThread
import sys
import cv2
import numpy as np

class videoThread(QThread):
    change_pixmap_signal=pyqtSignal(np.ndarray)

    def run(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, img = cap.read()
            img = cv2.resize(img, None, fx=0.5, fy=0.5)
            if ret:
                self.change_pixmap_signal.emit(img)
start = False
class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('demo')
        self.width = 640
        self.height = 480 
        self.setGeometry(10, 10, self.width, self.height)
        self.image_label = QLabel(self)
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

        self.vbox = QVBoxLayout()
        self.vbox.addWidget(self.image_label)
        self.vbox.addWidget(self.button1)
        self.vbox.addWidget(self.button2)

        self.setLayout(self.vbox)
        '''
        gray = QPixmap(w,h)
        gray.fill(QColor('darkGray'))
        self.image_label.setPixmap(gray)
        '''
        self.thread = videoThread()
        self.thread.change_pixmap_signal.connect(self.updateImage)
        self.thread.start()


    @pyqtSlot(np.ndarray)
    def updateImage(self, img):
        self.qtImg = self.convertCvQt(img)
        self.image_label.setPixmap(self.qtImg)

    def convertCvQt(self, img):
        rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgbImg.shape
        bytesPerLine = ch * w
        convertToQtFormat = QtGui.QImage(rgbImg.data, w, h, bytesPerLine, QtGui.QImage.Format_RGB888)
        p = convertToQtFormat.scaled(self.width, self.height, Qt.KeepAspectRatio)
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