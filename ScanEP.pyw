from PyQt5 import  QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIcon
from design import Ui_MainWindow

import sys
import functions

# Back up the reference to the exceptionhook
sys._excepthook = sys.excepthook

def my_exception_hook(exctype, value, traceback):
    # Print the error and traceback
    print(exctype, value, traceback)
    # Call the normal Exception hook after
    sys._excepthook(exctype, value, traceback)
    sys.exit(1)

# Set the exception hook to our wrapping function
sys.excepthook = my_exception_hook

class Second(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(Second, self).__init__(parent)

class scanEP(QtWidgets.QMainWindow):

    def __init__(self):
        super(scanEP, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.toolButton.clicked.connect(self.tb)
        self.ui.toolButton_2.clicked.connect(self.tb2)
        self.ui.toolButton_3.clicked.connect(self.tb3)
        self.ui.toolButton_4.clicked.connect(self.tb4)
        self.ui.toolButton_5.clicked.connect(self.tb5)
        self.ui.toolButton_6.clicked.connect(self.tb6)
        self.ui.toolButton_7.clicked.connect(self.tb7)
        self.ui.toolButton_8.clicked.connect(self.tb8)
        self.ui.toolButton_9.clicked.connect(self.tb9)
        self.ui.pushButton.clicked.connect(self.vecchanger)
        self.ui.pushButton_2.clicked.connect(self.methodchanger)
        self.ui.pushButton_3.clicked.connect(self.inpfilegenerator)
        self.ui.pushButton_4.clicked.connect(self.graph)
        self.ui.pushButton_5.clicked.connect(self.instructions)

    def tb(self):
        file1 = QtWidgets.QFileDialog.getOpenFileName(self, "Выберите расчетный файл")[0]
        self.ui.lineEdit.setText(str(file1))

    def tb2(self):
        file1 = QtWidgets.QFileDialog.getOpenFileName(self, "Выберите файл с новыми VEC")[0]
        self.ui.lineEdit_2.setText(str(file1))

    def tb3(self):
        file1 = QtWidgets.QFileDialog.getExistingDirectory(self, "Выберите директорию с входными файлами")
        self.ui.lineEdit_3.setText(str(file1))

    def tb4(self):
        file1 = QtWidgets.QFileDialog.getOpenFileName(self, "Выберите файл с новым методом")[0]
        self.ui.lineEdit_4.setText(str(file1))

    def tb5(self):
        file1 = QtWidgets.QFileDialog.getOpenFileName(self, "Выберите файл c настройками")[0]
        self.ui.lineEdit_5.setText(str(file1))

    def tb6(self):
        file1 = QtWidgets.QFileDialog.getOpenFileName(self, "Выберите расчетный файл 1")[0]
        self.ui.lineEdit_6.setText(str(file1))

    def tb7(self):
        file1 = QtWidgets.QFileDialog.getOpenFileName(self, "Выберите расчетный файл 2")[0]
        self.ui.lineEdit_7.setText(str(file1))

    def tb8(self):
        file1 = QtWidgets.QFileDialog.getExistingDirectory(self, "Выберите директорию для выходных файлов")
        self.ui.lineEdit_8.setText(str(file1))

    def tb9(self):
        file1 = QtWidgets.QFileDialog.getExistingDirectory(self, "Выберите папку c результатами QDPT")
        self.ui.lineEdit_9.setText(str(file1))

    def vecchanger(self):
        file1 = self.ui.lineEdit.text()
        file2 = self.ui.lineEdit_2.text()
        functions.vec_changer(file1, file2)
        self.ui.statusbar.showMessage('VEC group changes completed successfully')

    def methodchanger(self):
        directory = self.ui.lineEdit_3.text()
        file2 = self.ui.lineEdit_4.text()
        functions.method_changer(directory, file2)
        self.ui.statusbar.showMessage('Method changes completed successfully')

    def inpfilegenerator(self):
        file0 = self.ui.lineEdit_5.text()
        file1 = self.ui.lineEdit_6.text()
        file2 = self.ui.lineEdit_7.text()
        directory = self.ui.lineEdit_8.text()
        step = self.ui.doubleSpinBox.value()
        self.ui.statusbar.showMessage('Waiting')
        functions.file_generator(file0, file1, file2, directory, step)
        self.ui.statusbar.showMessage('Generation completed successfully')

    def graph(self):
        self.ui.statusbar.showMessage('Waiting')
        directory = self.ui.lineEdit_9.text()
        functions.qdptresult(directory)
        self.ui.statusbar.showMessage('Done')

    def instructions(self):
        ins = Second(self)
        ins.resize(750, 400)
        ins.move(300, 150)
        ins.setWindowTitle('Instructions')
        f = open('about.txt', 'r')
        text = f.read()
        info = QtWidgets.QLabel(text,self) # вынести текст инструкции в отдельный файл
        ins.setCentralWidget(info)
        f.close()
        ins.show()

try:
    sys.exit(app.exec_())
except:
    print("Working")

app = QtWidgets.QApplication([])
application = scanEP()
application.setWindowTitle('ScanEP')
application.setWindowIcon(QIcon('logo.png'))
application.show()
sys.exit(app.exec())
