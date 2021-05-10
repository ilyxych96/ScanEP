from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIcon
from design import Ui_MainWindow
from qdptplotter import Ui_Form
import sys
import os
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


class PlotterWindow(QtWidgets.QMainWindow):
    def __init__(self, parent):
        super(PlotterWindow, self).__init__(parent)
        self.qp = Ui_Form()
        self.qp.setupUi(self)
        self.qp.toolButton.clicked.connect(self.tb)
        self.qp.pushButton.clicked.connect(self.graph)

    def graph(self):
        directory = self.qp.lineEdit.text()

        state1 = self.qp.comboBox.currentText()
        state2 = self.qp.comboBox_2.currentText()

        linestyle1 = self.qp.comboBox_1.currentText()
        linewidth1 = self.qp.doubleSpinBox.value()
        linecolor1 = self.qp.comboBox_3.currentText()
        markerstyle1 = self.qp.comboBox_4.currentText()
        markersize1 = self.qp.comboBox_5.currentText()

        linestyle2 = self.qp.comboBox_6.currentText()
        linewidth2 = self.qp.doubleSpinBox_2.value()
        linecolor2 = self.qp.comboBox_8.currentText()
        markerstyle2 = self.qp.comboBox_9.currentText()
        markersize2 = self.qp.comboBox_10.currentText()

        axeswidth = self.qp.doubleSpinBox_7.value()
        ScaleFontsize = self.qp.doubleSpinBox_8.value()

        x_min = self.qp.doubleSpinBox_3.value()
        x_max = self.qp.doubleSpinBox_4.value()
        y_min = self.qp.doubleSpinBox_5.value()
        y_max = self.qp.doubleSpinBox_6.value()
        title = self.qp.lineEdit_2.text()
        if directory != 'Enter path to folder with QDPT .out files  --->':
            try:
                qdptouts = functions.qdptresult(directory, int(state1), int(state2), linestyle1, float(linewidth1),
                                        linecolor1, markerstyle1, markersize1, linestyle2, float(linewidth2),
                                        linecolor2, markerstyle2, markersize2, float(ScaleFontsize), float(axeswidth),
                                        float(x_min), float(x_max), float(y_min), float(y_max), title)
                self.qp.lineEdit_3.setText(str(qdptouts[0]))
                self.qp.lineEdit_4.setText(str(qdptouts[1]))
                self.qp.lineEdit_5.setText(str(qdptouts[2]))
            except:
                self.qp.lineEdit.setText('Error! Choose folder with results')

    def tb(self):
        file1 = QtWidgets.QFileDialog.getExistingDirectory(self, "Выберите папку c результатами QDPT")
        self.qp.lineEdit.setText(str(file1))


class scanEP(QtWidgets.QMainWindow):

    def instruction(self):
        os.path.join(os.getcwd(), 'about.txt')
        f = open(os.path.join(os.getcwd(), 'about.txt'), 'r')
        text = f.read()
        self.window2 = QtWidgets.QWidget()
        self.window2.setWindowIcon(QIcon('ico.png'))
        self.window2.setWindowTitle('Instruction')
        self.window2.label = QtWidgets.QLabel()
        self.window2.label.setText(text)
        self.window2.label.show()
        f.close()

    def qdpt(self):
        self.qp = PlotterWindow(self)
        self.qp.setWindowIcon(QIcon('ico.png'))
        self.qp.setWindowTitle('Energy profile plotter')
        self.qp.show()

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
        self.ui.toolButton_10.clicked.connect(self.tb10)
        self.ui.pushButton.clicked.connect(self.vecchanger)
        self.ui.pushButton_2.clicked.connect(self.methodchanger)
        self.ui.pushButton_3.clicked.connect(self.inpfilegenerator)
        self.ui.pushButton_6.clicked.connect(self.packdelimiter)
        self.ui.pushButton_4.clicked.connect(self.qdpt)
        self.ui.pushButton_5.clicked.connect(self.instruction)
        self.ui.pushButton_7.clicked.connect(self.openfolder)

    def openfolder(self):
        path = self.ui.lineEdit_8.text()
        mask = self.ui.lineEdit_10.text()
        if len(path) > 0:
            path = path + r'\\' + mask
            try:
                path = os.path.realpath(path)
                os.startfile(path)
            except:
                self.ui.statusbar.showMessage("Folder doesn't exist")
        else:
            self.ui.statusbar.showMessage("Folder doesn't exist")

    def tb(self):
        file1 = QtWidgets.QFileDialog.getExistingDirectory(self, "Выберите директории с расчетными файлами")
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
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Выберите директорию для выходных файлов")
        self.ui.lineEdit_8.setText(directory)

    def tb10(self):
        file1 = QtWidgets.QFileDialog.getOpenFileName(self, "Select pack file")[0]
        self.ui.lineEdit_11.setText(str(file1))

    def vecchanger(self):
        file1 = self.ui.lineEdit.text()
        file2 = self.ui.lineEdit_2.text()
        if len(file1) > 0 and len(file2) > 0:
            try:
                self.ui.statusbar.showMessage('In process')
                functions.vec_changer(file1, file2)
                self.ui.statusbar.showMessage('VEC group changes complete successfully')
            except:
                self.ui.statusbar.showMessage('Something wrong')
        else:
            self.ui.statusbar.showMessage('Some fields are empty')

    def methodchanger(self):
        directory = self.ui.lineEdit_3.text()
        file2 = self.ui.lineEdit_4.text()
        if len(file2) > 0 and len(directory) > 0:
            try:
                self.ui.statusbar.showMessage('In process')
                functions.method_changer(directory, file2)
                self.ui.statusbar.showMessage('Method changes complete successfully')
            except:
                self.ui.statusbar.showMessage('Something wrong')
        else:
            self.ui.statusbar.showMessage('Some fields are empty')

    def inpfilegenerator(self):
        file0 = self.ui.lineEdit_5.text()
        file1 = self.ui.lineEdit_6.text()
        file2 = self.ui.lineEdit_7.text()
        directory = self.ui.lineEdit_8.text()
        step = self.ui.doubleSpinBox.value()
        mask = self.ui.lineEdit_10.text()
        atomsinfirst = self.ui.comboBox.currentText()
        if not atomsinfirst:
            atomsinfirst = 'Dimer'

        if len(file0) > 0 and len(file1) > 0 and len(file2) > 0 and len(directory) > 0:
            try:
                self.ui.statusbar.showMessage('In process')
                functions.file_generator(file0, file1, file2, directory, step, mask, atomsinfirst)
                self.ui.statusbar.showMessage('Generation complete successfully')
            except:
                self.ui.statusbar.showMessage('Something wrong')
        else:
            self.ui.statusbar.showMessage('Some fields are empty')


    def packdelimiter(self):
        self.ui.statusbar.showMessage('In process')
        file0 = self.ui.lineEdit_11.text()
        minatomsinmolecule = self.ui.spinBox.value()
        maxcontact = self.ui.spinBox_2.value()
        if len(file0) > 0 and minatomsinmolecule >= 0 and maxcontact > 0:
            try:
                functions.packdelimiter(file0, minatomsinmolecule, maxcontact)
                self.ui.statusbar.showMessage('Separation complete successfully')
            except:
                self.ui.statusbar.showMessage('Something wrong')
        else:
            self.ui.statusbar.showMessage('Some fields are empty')


try:
    sys.exit(app.exec_())
except:
    print("ScanEp Working")


def main():
    app = QtWidgets.QApplication([])
    application = scanEP()
    application.setWindowTitle('ScanEP')
    application.setWindowIcon(QIcon('ico.png'))
    application.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
