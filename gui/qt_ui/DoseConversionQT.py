# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\Victor\Dropbox\DFR\film2dose\qt_ui\dose_conversion.ui'
#
# Created: Tue Sep 29 14:53:54 2015
#      by: pyside-uic 0.2.15 running on PySide 1.2.2
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui


class Ui_Form_film2dose(object):
    def setupUi(self, Form_film2dose):
        Form_film2dose.setObjectName("Form_film2dose")
        Form_film2dose.resize(1230, 651)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/icons/App_icon.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        Form_film2dose.setWindowIcon(icon)
        self.gridLayout_2 = QtGui.QGridLayout(Form_film2dose)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.lateral_checkbox = QtGui.QCheckBox(Form_film2dose)
        self.lateral_checkbox.setObjectName("lateral_checkbox")
        self.gridLayout_2.addWidget(self.lateral_checkbox, 1, 2, 1, 1)
        self.label = QtGui.QLabel(Form_film2dose)
        self.label.setObjectName("label")
        self.gridLayout_2.addWidget(self.label, 1, 1, 1, 1)
        self.to_dose = QtGui.QPushButton(Form_film2dose)
        self.to_dose.setIcon(icon)
        self.to_dose.setObjectName("to_dose")
        self.gridLayout_2.addWidget(self.to_dose, 2, 2, 1, 1)
        self.method_combo = QtGui.QComboBox(Form_film2dose)
        self.method_combo.setObjectName("method_combo")
        self.method_combo.addItem("")
        self.method_combo.addItem("")
        self.method_combo.addItem("")
        self.gridLayout_2.addWidget(self.method_combo, 2, 1, 1, 1)
        self.import_image = QtGui.QPushButton(Form_film2dose)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/icons/Import Picture Document.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.import_image.setIcon(icon1)
        self.import_image.setObjectName("import_image")
        self.gridLayout_2.addWidget(self.import_image, 0, 2, 1, 1)
        self.calib_button = QtGui.QPushButton(Form_film2dose)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/icons/curvechart-edit.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.calib_button.setIcon(icon2)
        self.calib_button.setObjectName("calib_button")
        self.gridLayout_2.addWidget(self.calib_button, 0, 1, 1, 1)
        self.dose_widget = QtGui.QWidget(Form_film2dose)
        self.dose_widget.setObjectName("dose_widget")
        self.gridLayout_2.addWidget(self.dose_widget, 3, 2, 1, 1)
        self.image_widget = QtGui.QWidget(Form_film2dose)
        self.image_widget.setObjectName("image_widget")
        self.gridLayout_2.addWidget(self.image_widget, 3, 1, 1, 1)

        self.retranslateUi(Form_film2dose)
        QtCore.QMetaObject.connectSlotsByName(Form_film2dose)

    def retranslateUi(self, Form_film2dose):
        Form_film2dose.setWindowTitle(
            QtGui.QApplication.translate("Form_film2dose", "Dose Conversion", None, QtGui.QApplication.UnicodeUTF8))
        self.lateral_checkbox.setText(
            QtGui.QApplication.translate("Form_film2dose", "Lateral Correction", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("Form_film2dose",
                                                        "<html><head/><body><p align=\"center\"><span style=\" font-weight:600;\">Calculation Mehod</span></p></body></html>",
                                                        None, QtGui.QApplication.UnicodeUTF8))
        self.to_dose.setText(
            QtGui.QApplication.translate("Form_film2dose", "Film2Dose", None, QtGui.QApplication.UnicodeUTF8))
        self.method_combo.setItemText(0, QtGui.QApplication.translate("Form_film2dose", "Single Channel", None,
                                                                      QtGui.QApplication.UnicodeUTF8))
        self.method_combo.setItemText(1, QtGui.QApplication.translate("Form_film2dose", "Robust Average", None,
                                                                      QtGui.QApplication.UnicodeUTF8))
        self.method_combo.setItemText(2, QtGui.QApplication.translate("Form_film2dose", "Robust Multichannel", None,
                                                                      QtGui.QApplication.UnicodeUTF8))
        self.import_image.setText(
            QtGui.QApplication.translate("Form_film2dose", "Import Image", None, QtGui.QApplication.UnicodeUTF8))
        self.calib_button.setText(QtGui.QApplication.translate("Form_film2dose", "Import Calibration Data", None,
                                                               QtGui.QApplication.UnicodeUTF8))
