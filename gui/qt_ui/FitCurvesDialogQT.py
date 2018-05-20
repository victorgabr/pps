# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\Victor\Dropbox\DFR\film2dose\qt_ui\fit_curves_dialog.ui'
#
# Created: Tue Sep 29 14:53:39 2015
#      by: pyside-uic 0.2.15 running on PySide 1.2.2
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1065, 391)
        self.horizontalLayout = QtGui.QHBoxLayout(Dialog)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout_8 = QtGui.QVBoxLayout()
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.horizontalLayout_5 = QtGui.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.comboBox = QtGui.QComboBox(Dialog)
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.horizontalLayout_5.addWidget(self.comboBox)
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem)
        self.radioButton_eq1 = QtGui.QRadioButton(Dialog)
        self.radioButton_eq1.setObjectName("radioButton_eq1")
        self.horizontalLayout_5.addWidget(self.radioButton_eq1)
        self.RadioButton_eq2 = QtGui.QRadioButton(Dialog)
        self.RadioButton_eq2.setObjectName("RadioButton_eq2")
        self.horizontalLayout_5.addWidget(self.RadioButton_eq2)
        self.radioButton_3 = QtGui.QRadioButton(Dialog)
        self.radioButton_3.setObjectName("radioButton_3")
        self.horizontalLayout_5.addWidget(self.radioButton_3)
        spacerItem1 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem1)
        self.select_curve = QtGui.QPushButton(Dialog)
        self.select_curve.setObjectName("select_curve")
        self.horizontalLayout_5.addWidget(self.select_curve)
        self.finish_button = QtGui.QPushButton(Dialog)
        self.finish_button.setObjectName("finish_button")
        self.horizontalLayout_5.addWidget(self.finish_button)
        self.verticalLayout_8.addLayout(self.horizontalLayout_5)
        self.horizontalLayout.addLayout(self.verticalLayout_8)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QtGui.QApplication.translate("Dialog", "Dialog", None, QtGui.QApplication.UnicodeUTF8))
        self.comboBox.setItemText(0, QtGui.QApplication.translate("Dialog", "Red Channel", None,
                                                                  QtGui.QApplication.UnicodeUTF8))
        self.comboBox.setItemText(1, QtGui.QApplication.translate("Dialog", "Green Channel", None,
                                                                  QtGui.QApplication.UnicodeUTF8))
        self.comboBox.setItemText(2, QtGui.QApplication.translate("Dialog", "Blue Channel", None,
                                                                  QtGui.QApplication.UnicodeUTF8))
        self.radioButton_eq1.setText(
            QtGui.QApplication.translate("Dialog", "Equation 1", None, QtGui.QApplication.UnicodeUTF8))
        self.RadioButton_eq2.setText(
            QtGui.QApplication.translate("Dialog", "Equation 2", None, QtGui.QApplication.UnicodeUTF8))
        self.radioButton_3.setText(
            QtGui.QApplication.translate("Dialog", "Equation 3", None, QtGui.QApplication.UnicodeUTF8))
        self.select_curve.setText(
            QtGui.QApplication.translate("Dialog", "Select", None, QtGui.QApplication.UnicodeUTF8))
        self.finish_button.setText(
            QtGui.QApplication.translate("Dialog", "Finish", None, QtGui.QApplication.UnicodeUTF8))
