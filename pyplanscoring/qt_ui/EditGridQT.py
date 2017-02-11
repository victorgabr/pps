# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\Victor\Dropbox\DFR\film2dose\qt_ui\edit_grid.ui'
#
# Created: Tue Sep 29 14:53:43 2015
#      by: pyside-uic 0.2.15 running on PySide 1.2.2
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(392, 125)
        self.gridLayout = QtGui.QGridLayout(Dialog)
        self.gridLayout.setObjectName("gridLayout")
        self.ny_spin = QtGui.QSpinBox(Dialog)
        self.ny_spin.setObjectName("ny_spin")
        self.gridLayout.addWidget(self.ny_spin, 4, 0, 1, 1)
        self.label = QtGui.QLabel(Dialog)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 3, 0, 1, 1)
        self.label_2 = QtGui.QLabel(Dialog)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 3, 1, 1, 1)
        self.label_4 = QtGui.QLabel(Dialog)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 0, 1, 1, 1)
        self.label_3 = QtGui.QLabel(Dialog)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 0, 0, 1, 1)
        self.yd_spin = QtGui.QDoubleSpinBox(Dialog)
        self.yd_spin.setProperty("value", 0.0)
        self.yd_spin.setObjectName("yd_spin")
        self.gridLayout.addWidget(self.yd_spin, 4, 1, 1, 1)
        self.buttonBox = QtGui.QDialogButtonBox(Dialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel | QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayout.addWidget(self.buttonBox, 1, 3, 1, 1)
        self.nx_spin = QtGui.QSpinBox(Dialog)
        self.nx_spin.setProperty("value", 0)
        self.nx_spin.setObjectName("nx_spin")
        self.gridLayout.addWidget(self.nx_spin, 1, 0, 1, 1)
        self.xd_spin = QtGui.QDoubleSpinBox(Dialog)
        self.xd_spin.setSingleStep(1.0)
        self.xd_spin.setProperty("value", 0.0)
        self.xd_spin.setObjectName("xd_spin")
        self.gridLayout.addWidget(self.xd_spin, 1, 1, 1, 1)

        self.retranslateUi(Dialog)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL("accepted()"), Dialog.accept)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL("rejected()"), Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QtGui.QApplication.translate("Dialog", "Dialog", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("Dialog", "y points", None, QtGui.QApplication.UnicodeUTF8))
        self.label_2.setText(
            QtGui.QApplication.translate("Dialog", "y spacing (mm)", None, QtGui.QApplication.UnicodeUTF8))
        self.label_4.setText(
            QtGui.QApplication.translate("Dialog", "x spacing (mm)", None, QtGui.QApplication.UnicodeUTF8))
        self.label_3.setText(QtGui.QApplication.translate("Dialog", "x points", None, QtGui.QApplication.UnicodeUTF8))
