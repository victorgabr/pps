# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\vgalves\Dropbox\DFR\film2dose\qt_ui\DoseCompAbs.ui'
#
# Created: Mon Nov 23 16:50:43 2015
#      by: pyside-uic 0.2.15 running on PySide 1.2.2
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui


class Ui_DoseCompAbs(object):
    def setupUi(self, DoseCompAbs):
        DoseCompAbs.setObjectName("DoseCompAbs")
        DoseCompAbs.resize(800, 590)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/icons/compare.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        DoseCompAbs.setWindowIcon(icon)
        self.verticalLayout = QtGui.QVBoxLayout(DoseCompAbs)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.abs_radio = QtGui.QRadioButton(DoseCompAbs)
        self.abs_radio.setObjectName("abs_radio")
        self.horizontalLayout.addWidget(self.abs_radio)
        self.rel_radio = QtGui.QRadioButton(DoseCompAbs)
        self.rel_radio.setChecked(True)
        self.rel_radio.setObjectName("rel_radio")
        self.horizontalLayout.addWidget(self.rel_radio)
        self.open_images = QtGui.QPushButton(DoseCompAbs)
        self.open_images.setObjectName("open_images")
        self.horizontalLayout.addWidget(self.open_images)
        self.profiles_button = QtGui.QPushButton(DoseCompAbs)
        self.profiles_button.setObjectName("profiles_button")
        self.horizontalLayout.addWidget(self.profiles_button)
        self.compare_button = QtGui.QPushButton(DoseCompAbs)
        self.compare_button.setObjectName("compare_button")
        self.horizontalLayout.addWidget(self.compare_button)
        self.save_images = QtGui.QPushButton(DoseCompAbs)
        self.save_images.setObjectName("save_images")
        self.horizontalLayout.addWidget(self.save_images)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.gridLayout = QtGui.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.dose_diff_widget = QtGui.QWidget(DoseCompAbs)
        self.dose_diff_widget.setObjectName("dose_diff_widget")
        self.gridLayout.addWidget(self.dose_diff_widget, 1, 0, 1, 1)
        self.hist_widget = QtGui.QWidget(DoseCompAbs)
        self.hist_widget.setObjectName("hist_widget")
        self.gridLayout.addWidget(self.hist_widget, 1, 1, 1, 1)
        self.film_widget = QtGui.QWidget(DoseCompAbs)
        self.film_widget.setObjectName("film_widget")
        self.gridLayout.addWidget(self.film_widget, 0, 1, 1, 1)
        self.tps_widget = QtGui.QWidget(DoseCompAbs)
        self.tps_widget.setObjectName("tps_widget")
        self.gridLayout.addWidget(self.tps_widget, 0, 0, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)

        self.retranslateUi(DoseCompAbs)
        QtCore.QMetaObject.connectSlotsByName(DoseCompAbs)

    def retranslateUi(self, DoseCompAbs):
        DoseCompAbs.setWindowTitle(QtGui.QApplication.translate("DoseCompAbs", "Film2Dose - Dose Difference ", None,
                                                                QtGui.QApplication.UnicodeUTF8))
        self.abs_radio.setText(
            QtGui.QApplication.translate("DoseCompAbs", "Absolute Difference", None, QtGui.QApplication.UnicodeUTF8))
        self.rel_radio.setText(QtGui.QApplication.translate("DoseCompAbs", "Relative Difference (%)", None,
                                                            QtGui.QApplication.UnicodeUTF8))
        self.open_images.setText(
            QtGui.QApplication.translate("DoseCompAbs", "Open Images", None, QtGui.QApplication.UnicodeUTF8))
        self.profiles_button.setText(
            QtGui.QApplication.translate("DoseCompAbs", "Profiles", None, QtGui.QApplication.UnicodeUTF8))
        self.compare_button.setText(
            QtGui.QApplication.translate("DoseCompAbs", "Dose ", None, QtGui.QApplication.UnicodeUTF8))
        self.save_images.setText(
            QtGui.QApplication.translate("DoseCompAbs", "Save Images", None, QtGui.QApplication.UnicodeUTF8))
