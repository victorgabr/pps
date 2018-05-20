# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '/home/victor/Dropbox/DFR/film2dose/qt_ui/DoseCompMethod.ui'
#
# Created: Sun Oct  4 13:33:46 2015
#      by: pyside-uic 0.2.15 running on PySide 1.2.2
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(400, 114)
        self.gridLayout = QtGui.QGridLayout(Form)
        self.gridLayout.setObjectName("gridLayout")
        self.dose_diff_button = QtGui.QPushButton(Form)
        self.dose_diff_button.setObjectName("dose_diff_button")
        self.gridLayout.addWidget(self.dose_diff_button, 0, 0, 1, 1)
        self.gamma_button = QtGui.QPushButton(Form)
        self.gamma_button.setObjectName("gamma_button")
        self.gridLayout.addWidget(self.gamma_button, 0, 1, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(QtGui.QApplication.translate("Form", "Select method", None, QtGui.QApplication.UnicodeUTF8))
        self.dose_diff_button.setText(
            QtGui.QApplication.translate("Form", "Dose Difference", None, QtGui.QApplication.UnicodeUTF8))
        self.gamma_button.setText(
            QtGui.QApplication.translate("Form", "Gamma Index", None, QtGui.QApplication.UnicodeUTF8))
