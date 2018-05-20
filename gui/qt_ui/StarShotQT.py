# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\Victor\Dropbox\DFR\film2dose\qt_ui\starshot.ui'
#
# Created: Tue Sep 29 14:53:57 2015
#      by: pyside-uic 0.2.15 running on PySide 1.2.2
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui


class Ui_StarShotWidget(object):
    def setupUi(self, StarShotWidget):
        StarShotWidget.setObjectName("StarShotWidget")
        StarShotWidget.resize(1047, 710)
        self.verticalLayoutWidget = QtGui.QWidget(StarShotWidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(9, 16, 1031, 701))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtGui.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.gridLayout = QtGui.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.auto_radio = QtGui.QRadioButton(self.verticalLayoutWidget)
        self.auto_radio.setObjectName("auto_radio")
        self.gridLayout.addWidget(self.auto_radio, 0, 0, 1, 1)
        self.manual_radio = QtGui.QRadioButton(self.verticalLayoutWidget)
        self.manual_radio.setObjectName("manual_radio")
        self.gridLayout.addWidget(self.manual_radio, 0, 1, 1, 1)
        self.analyse_button = QtGui.QPushButton(self.verticalLayoutWidget)
        self.analyse_button.setObjectName("analyse_button")
        self.gridLayout.addWidget(self.analyse_button, 0, 2, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)

        self.retranslateUi(StarShotWidget)
        QtCore.QMetaObject.connectSlotsByName(StarShotWidget)

    def retranslateUi(self, StarShotWidget):
        StarShotWidget.setWindowTitle(
            QtGui.QApplication.translate("StarShotWidget", "Form", None, QtGui.QApplication.UnicodeUTF8))
        self.auto_radio.setText(
            QtGui.QApplication.translate("StarShotWidget", "Auto  center", None, QtGui.QApplication.UnicodeUTF8))
        self.manual_radio.setText(
            QtGui.QApplication.translate("StarShotWidget", "Manual  center", None, QtGui.QApplication.UnicodeUTF8))
        self.analyse_button.setText(
            QtGui.QApplication.translate("StarShotWidget", "Analyse", None, QtGui.QApplication.UnicodeUTF8))
