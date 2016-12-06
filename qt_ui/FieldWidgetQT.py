# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\vgalves\Dropbox\DFR\film2dose\qt_ui\field_widget.ui'
#
# Created: Thu Oct 15 17:50:22 2015
#      by: pyside-uic 0.2.15 running on PySide 1.2.2
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(673, 414)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/icons/Ruler.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        Form.setWindowIcon(icon)
        self.gridLayout = QtGui.QGridLayout(Form)
        self.gridLayout.setObjectName("gridLayout")
        self.field_widget = QtGui.QWidget(Form)
        self.field_widget.setObjectName("field_widget")
        self.gridLayout.addWidget(self.field_widget, 1, 0, 1, 3)
        self.analyse_button = QtGui.QPushButton(Form)
        self.analyse_button.setObjectName("analyse_button")
        self.gridLayout.addWidget(self.analyse_button, 0, 2, 1, 1)
        self.open_button = QtGui.QPushButton(Form)
        self.open_button.setObjectName("open_button")
        self.gridLayout.addWidget(self.open_button, 0, 0, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(
            QtGui.QApplication.translate("Form", "Symmetry and Flatness tool", None, QtGui.QApplication.UnicodeUTF8))
        self.analyse_button.setText(
            QtGui.QApplication.translate("Form", "Analyse", None, QtGui.QApplication.UnicodeUTF8))
        self.open_button.setText(
            QtGui.QApplication.translate("Form", "Open Image", None, QtGui.QApplication.UnicodeUTF8))


from film2dose.qt_ui import icons_rc
