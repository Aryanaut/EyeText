from PyQt5.QtWidgets import *

def ifClicked():
    print("Clicked")

def ifClicked2():
    print("Other clicked")

app = QApplication([])
window = QMainWindow()

centerWidget = QWidget()
vbox = QVBoxLayout()
button = QPushButton('Test', centerWidget)
button2 = QPushButton('Test2', centerWidget)
button.setGeometry(0,50,120,40)
button.clicked.connect(ifClicked)
button2.clicked.connect(ifClicked2)
vbox.addWidget(button)
vbox.addWidget(button2)
# window.setCentralWidget(button)
window.setWindowTitle('mainwindow')
window.show()

app.exit(app.exec_())