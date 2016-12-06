# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\Victor\Dropbox\DFR\film2dose\qt_ui\fit_mode_widget.ui'
#
# Created: Tue Sep 29 14:54:05 2015
#      by: pyside-uic 0.2.15 running on PySide 1.2.2
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui


class Ui_FitMode(object):
    def setupUi(self, FitMode):
        FitMode.setObjectName("FitMode")
        FitMode.resize(1061, 85)
        self.gridLayout = QtGui.QGridLayout(FitMode)
        self.gridLayout.setObjectName("gridLayout")
        self.load_button = QtGui.QPushButton(FitMode)
        self.load_button.setObjectName("load_button")
        self.gridLayout.addWidget(self.load_button, 0, 0, 1, 1)
        self.fit_channels_button = QtGui.QPushButton(FitMode)
        self.fit_channels_button.setObjectName("fit_channels_button")
        self.gridLayout.addWidget(self.fit_channels_button, 1, 0, 1, 1)
        self.evo_button = QtGui.QPushButton(FitMode)
        self.evo_button.setObjectName("evo_button")
        self.gridLayout.addWidget(self.evo_button, 1, 1, 1, 1)
        self.save_cal_button = QtGui.QPushButton(FitMode)
        self.save_cal_button.setObjectName("save_cal_button")
        self.gridLayout.addWidget(self.save_cal_button, 0, 1, 1, 1)

        self.retranslateUi(FitMode)
        QtCore.QMetaObject.connectSlotsByName(FitMode)

    def retranslateUi(self, FitMode):
        FitMode.setWindowTitle(
            QtGui.QApplication.translate("FitMode", "Calibration Options", None, QtGui.QApplication.UnicodeUTF8))
        self.load_button.setText(
            QtGui.QApplication.translate("FitMode", "Load calibration data", None, QtGui.QApplication.UnicodeUTF8))
        self.fit_channels_button.setText(
            QtGui.QApplication.translate("FitMode", "Fit channels", None, QtGui.QApplication.UnicodeUTF8))
        self.evo_button.setText(
            QtGui.QApplication.translate("FitMode", "Evolutionary Optimization", None, QtGui.QApplication.UnicodeUTF8))
        self.save_cal_button.setText(
            QtGui.QApplication.translate("FitMode", "Save Calibration", None, QtGui.QApplication.UnicodeUTF8))
