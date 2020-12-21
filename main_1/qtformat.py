from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QHBoxLayout, QPushButton
from PyQt5.QtGui import QPixmap, QImage, QColor
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QObject, QThread
import sys
import cv2
import numpy as np


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('WindowTitle')
        self.width, self.height = (640, 480)
        self.setGeometry(10, 10, self.width, self.height)
        self.image_label = QLabel(self)
        
        # adding widgets to the main window
        self.vbox = QVBoxLayout()
        self.vbox.addWidget(self.image_label)
        self.setLayout(self.vbox)

    # @pyqtSlot #uncomment for video
    def covertCvQt(self,img):
        rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgbImg.shape
        bytesPerLine = ch * w
        convertToQtFormat = QtGui.QImage(rgbImg.data, w, h, bytesPerLine, QtGui.QImage.Format_RGB888)
        p = convertToQtFormat(self.width, self.height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec_())
