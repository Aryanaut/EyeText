from PyQt5 import QtWidgets, uic
import sys

class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('basic.ui', self)

        self.button = self.findChild(QtWidgets.QPushButton, 'button') # Find the button
        self.button.clicked.connect(self.printButtonPressed) # Remember to pass the definition/method, not the return value!

        self.input = self.findChild(QtWidgets.QLineEdit, 'input')

        self.show()

    def printButtonPressed(self):
        # This is executed when the button is pressed
        print('ok')

app = QtWidgets.QApplication(sys.argv)
window = Ui()
app.exec_()