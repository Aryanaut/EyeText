from PyQt5.QtWidgets import QApplication, QMainWindow, QSlider, QVBoxLayout
from PyQt5.QtCore import Qt

def changeValue(value):
    x = value
    print(x)

app = QApplication([])
window = QMainWindow()

slider = QSlider(Qt.Horizontal, window)
slider.valueChanged[int].connect(changeValue)
slider.setMinimum(0)
slider.setMaximum(255)

vbox = QVBoxLayout()
vbox.addWidget(slider)

window.setWindowTitle('trial')
window.setLayout(vbox)
window.show()

app.exit(app.exec_())