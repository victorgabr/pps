# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\Victor\Dropbox\DFR\film2dose\qt_ui\get_cal_points.ui'
#
# Created: Tue Sep 29 14:54:15 2015
#      by: pyside-uic 0.2.15 running on PySide 1.2.2
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui


class Ui_GetCalPoints(object):
    def setupUi(self, GetCalPoints):
        GetCalPoints.setObjectName("GetCalPoints")
        GetCalPoints.resize(988, 681)
        self.gridLayout_2 = QtGui.QGridLayout(GetCalPoints)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.film_od_widget = QtGui.QWidget(GetCalPoints)
        self.film_od_widget.setObjectName("film_od_widget")
        self.gridLayout_2.addWidget(self.film_od_widget, 5, 2, 1, 1)
        self.reference_widget = QtGui.QWidget(GetCalPoints)
        self.reference_widget.setObjectName("reference_widget")
        self.gridLayout_2.addWidget(self.reference_widget, 5, 1, 1, 1)
        self.load_data_button = QtGui.QPushButton(GetCalPoints)
        self.load_data_button.setObjectName("load_data_button")
        self.gridLayout_2.addWidget(self.load_data_button, 2, 1, 1, 1)
        self.get_points_Button = QtGui.QPushButton(GetCalPoints)
        self.get_points_Button.setObjectName("get_points_Button")
        self.gridLayout_2.addWidget(self.get_points_Button, 4, 1, 1, 1)
        self.register_button = QtGui.QPushButton(GetCalPoints)
        self.register_button.setObjectName("register_button")
        self.gridLayout_2.addWidget(self.register_button, 2, 2, 1, 1)
        self.point_radio = QtGui.QRadioButton(GetCalPoints)
        self.point_radio.setChecked(True)
        self.point_radio.setObjectName("point_radio")
        self.gridLayout_2.addWidget(self.point_radio, 3, 1, 1, 1)
        self.grid_radio = QtGui.QRadioButton(GetCalPoints)
        self.grid_radio.setChecked(False)
        self.grid_radio.setObjectName("grid_radio")
        self.gridLayout_2.addWidget(self.grid_radio, 3, 2, 1, 1)
        self.save_button = QtGui.QPushButton(GetCalPoints)
        self.save_button.setObjectName("save_button")
        self.gridLayout_2.addWidget(self.save_button, 4, 2, 1, 1)
        self.result_widget = QtGui.QWidget(GetCalPoints)
        self.result_widget.setObjectName("result_widget")
        self.gridLayout_2.addWidget(self.result_widget, 6, 2, 1, 1)
        self.table_widget = QtGui.QWidget(GetCalPoints)
        self.table_widget.setObjectName("table_widget")
        self.gridLayout_2.addWidget(self.table_widget, 6, 1, 1, 1)

        self.retranslateUi(GetCalPoints)
        QtCore.QMetaObject.connectSlotsByName(GetCalPoints)

    def retranslateUi(self, GetCalPoints):
        GetCalPoints.setWindowTitle(
            QtGui.QApplication.translate("GetCalPoints", "Select calibration data acquisition mode", None,
                                         QtGui.QApplication.UnicodeUTF8))
        self.load_data_button.setText(
            QtGui.QApplication.translate("GetCalPoints", "Load data", None, QtGui.QApplication.UnicodeUTF8))
        self.get_points_Button.setText(
            QtGui.QApplication.translate("GetCalPoints", "get points", None, QtGui.QApplication.UnicodeUTF8))
        self.register_button.setText(
            QtGui.QApplication.translate("GetCalPoints", "Register Images", None, QtGui.QApplication.UnicodeUTF8))
        self.point_radio.setText(QtGui.QApplication.translate("GetCalPoints", "Point based acquisition", None,
                                                              QtGui.QApplication.UnicodeUTF8))
        self.grid_radio.setText(QtGui.QApplication.translate("GetCalPoints", "Grid based acquisition", None,
                                                             QtGui.QApplication.UnicodeUTF8))
        self.save_button.setText(
            QtGui.QApplication.translate("GetCalPoints", "Save calibration data", None, QtGui.QApplication.UnicodeUTF8))
