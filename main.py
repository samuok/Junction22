from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
from PyQt5.QtCore import Qt

class Ui(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        loadUi("mainscreen.ui",self)
        self.Answer1.valueChanged.connect(self.setValue1)
        self.Answer2.valueChanged.connect(self.setValue2)
        self.Answer3.valueChanged.connect(self.setValue3)
        self.Answer4.valueChanged.connect(self.setValue4)
        self.Submit.clicked.connect(self.loadtoDataBase)

    def setValue1(self, item):
        self.Value1.setText(str(item))

    def setValue2(self, item):
        self.Value2.setText(str(item))

    def setValue3(self, item):
        self.Value3.setText(str(item))

    def setValue4(self, item):
        self.Value4.setText(str(item))

    def loadtoDataBase(self):
        print("hello")

app = QApplication([])
window = Ui()
window.show()
app.exec_()

