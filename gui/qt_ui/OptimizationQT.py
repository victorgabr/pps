# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\Victor\Dropbox\DFR\film2dose\qt_ui\evo_widget.ui'
#
# Created: Tue Sep 29 14:54:23 2015
#      by: pyside-uic 0.2.15 running on PySide 1.2.2
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1161, 691)
        self.gridLayout = QtGui.QGridLayout(Form)
        self.gridLayout.setObjectName("gridLayout")
        self.optimize_button = QtGui.QPushButton(Form)
        self.optimize_button.setObjectName("optimize_button")
        self.gridLayout.addWidget(self.optimize_button, 12, 1, 1, 1)
        self.pop_spin = QtGui.QSpinBox(Form)
        self.pop_spin.setMaximum(5000)
        self.pop_spin.setProperty("value", 200)
        self.pop_spin.setObjectName("pop_spin")
        self.gridLayout.addWidget(self.pop_spin, 7, 0, 1, 1)
        self.label_5 = QtGui.QLabel(Form)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 6, 0, 1, 1)
        self.crop_border_spin = QtGui.QDoubleSpinBox(Form)
        self.crop_border_spin.setSingleStep(0.1)
        self.crop_border_spin.setProperty("value", 5.0)
        self.crop_border_spin.setObjectName("crop_border_spin")
        self.gridLayout.addWidget(self.crop_border_spin, 4, 1, 1, 1)
        self.label_2 = QtGui.QLabel(Form)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 3, 0, 1, 1)
        self.eq_combo = QtGui.QComboBox(Form)
        self.eq_combo.setObjectName("eq_combo")
        self.eq_combo.addItem("")
        self.eq_combo.addItem("")
        self.eq_combo.addItem("")
        self.eq_combo.addItem("")
        self.gridLayout.addWidget(self.eq_combo, 9, 0, 1, 1)
        self.mode_combo = QtGui.QComboBox(Form)
        self.mode_combo.setObjectName("mode_combo")
        self.mode_combo.addItem("")
        self.mode_combo.addItem("")
        self.mode_combo.addItem("")
        self.gridLayout.addWidget(self.mode_combo, 9, 1, 1, 1)
        self.label_8 = QtGui.QLabel(Form)
        self.label_8.setObjectName("label_8")
        self.gridLayout.addWidget(self.label_8, 2, 0, 1, 1)
        self.setup_button = QtGui.QPushButton(Form)
        self.setup_button.setObjectName("setup_button")
        self.gridLayout.addWidget(self.setup_button, 12, 0, 1, 1)
        self.label_4 = QtGui.QLabel(Form)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 8, 0, 1, 1)
        self.color_combo = QtGui.QComboBox(Form)
        self.color_combo.setObjectName("color_combo")
        self.color_combo.addItem("")
        self.color_combo.addItem("")
        self.color_combo.addItem("")
        self.gridLayout.addWidget(self.color_combo, 2, 1, 1, 1)
        self.label_6 = QtGui.QLabel(Form)
        self.label_6.setObjectName("label_6")
        self.gridLayout.addWidget(self.label_6, 6, 1, 1, 1)
        self.pixel_size_spin = QtGui.QDoubleSpinBox(Form)
        self.pixel_size_spin.setProperty("value", 1.0)
        self.pixel_size_spin.setObjectName("pixel_size_spin")
        self.gridLayout.addWidget(self.pixel_size_spin, 4, 0, 1, 1)
        self.label = QtGui.QLabel(Form)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 3, 1, 1, 1)
        self.poly_range_spin = QtGui.QDoubleSpinBox(Form)
        self.poly_range_spin.setMaximum(1000000.0)
        self.poly_range_spin.setProperty("value", 1.0)
        self.poly_range_spin.setObjectName("poly_range_spin")
        self.gridLayout.addWidget(self.poly_range_spin, 7, 1, 1, 1)
        self.label_3 = QtGui.QLabel(Form)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 8, 1, 1, 1)
        self.save_cal = QtGui.QPushButton(Form)
        self.save_cal.setObjectName("save_cal")
        self.gridLayout.addWidget(self.save_cal, 13, 1, 1, 1)
        self.label_7 = QtGui.QLabel(Form)
        self.label_7.setObjectName("label_7")
        self.gridLayout.addWidget(self.label_7, 14, 0, 1, 1)
        self.seed_spin = QtGui.QSpinBox(Form)
        self.seed_spin.setProperty("value", 1)
        self.seed_spin.setObjectName("seed_spin")
        self.gridLayout.addWidget(self.seed_spin, 14, 1, 1, 1)
        self.image_widget = QtGui.QWidget(Form)
        self.image_widget.setObjectName("image_widget")
        self.gridLayout.addWidget(self.image_widget, 10, 1, 1, 1)
        self.ref_widget = QtGui.QWidget(Form)
        self.ref_widget.setObjectName("ref_widget")
        self.gridLayout.addWidget(self.ref_widget, 10, 0, 1, 1)
        self.bg_checkBox = QtGui.QCheckBox(Form)
        self.bg_checkBox.setChecked(False)
        self.bg_checkBox.setObjectName("bg_checkBox")
        self.gridLayout.addWidget(self.bg_checkBox, 1, 0, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(QtGui.QApplication.translate("Form", "Optimization ", None, QtGui.QApplication.UnicodeUTF8))
        self.optimize_button.setText(
            QtGui.QApplication.translate("Form", "Optimize", None, QtGui.QApplication.UnicodeUTF8))
        self.label_5.setText(QtGui.QApplication.translate("Form",
                                                          "<html><head/><body><p align=\"center\"><span style=\" font-weight:600;\">Population size</span></p></body></html>",
                                                          None, QtGui.QApplication.UnicodeUTF8))
        self.label_2.setText(QtGui.QApplication.translate("Form",
                                                          "<html><head/><body><p align=\"center\"><span style=\" font-weight:600;\">Optimization pixel size (mm)</span></p></body></html>",
                                                          None, QtGui.QApplication.UnicodeUTF8))
        self.eq_combo.setItemText(0, QtGui.QApplication.translate("Form", "Equation 1 - Inverse Log poly", None,
                                                                  QtGui.QApplication.UnicodeUTF8))
        self.eq_combo.setItemText(1, QtGui.QApplication.translate("Form", "Equation 2 - Inverse poly", None,
                                                                  QtGui.QApplication.UnicodeUTF8))
        self.eq_combo.setItemText(2, QtGui.QApplication.translate("Form", "Equation 3 - Inverse arctan poly", None,
                                                                  QtGui.QApplication.UnicodeUTF8))
        self.eq_combo.setItemText(3, QtGui.QApplication.translate("Form", "Equation 4 - 4th Degree Poly", None,
                                                                  QtGui.QApplication.UnicodeUTF8))
        self.mode_combo.setItemText(0, QtGui.QApplication.translate("Form", "Polynomial curve fitting ", None,
                                                                    QtGui.QApplication.UnicodeUTF8))
        self.mode_combo.setItemText(1, QtGui.QApplication.translate("Form", "Lateral correction", None,
                                                                    QtGui.QApplication.UnicodeUTF8))
        self.mode_combo.setItemText(2, QtGui.QApplication.translate("Form", "Poly fit and correction", None,
                                                                    QtGui.QApplication.UnicodeUTF8))
        self.label_8.setText(QtGui.QApplication.translate("Form",
                                                          "<html><head/><body><p align=\"right\"><span style=\" font-weight:600;\">Color Channel:</span></p></body></html>",
                                                          None, QtGui.QApplication.UnicodeUTF8))
        self.setup_button.setText(
            QtGui.QApplication.translate("Form", "Setup optimization", None, QtGui.QApplication.UnicodeUTF8))
        self.label_4.setText(QtGui.QApplication.translate("Form",
                                                          "<html><head/><body><p align=\"center\"><span style=\" font-weight:600;\">Select Equation</span></p></body></html>",
                                                          None, QtGui.QApplication.UnicodeUTF8))
        self.color_combo.setItemText(0,
                                     QtGui.QApplication.translate("Form", "Red", None, QtGui.QApplication.UnicodeUTF8))
        self.color_combo.setItemText(1, QtGui.QApplication.translate("Form", "Green", None,
                                                                     QtGui.QApplication.UnicodeUTF8))
        self.color_combo.setItemText(2,
                                     QtGui.QApplication.translate("Form", "Blue", None, QtGui.QApplication.UnicodeUTF8))
        self.label_6.setText(QtGui.QApplication.translate("Form",
                                                          "<html><head/><body><p align=\"center\"><span style=\" font-weight:600;\">Poly bounds (+-)</span></p></body></html>",
                                                          None, QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("Form",
                                                        "<html><head/><body><p align=\"center\"><span style=\" font-weight:600;\">Crop border (mm)</span></p></body></html>",
                                                        None, QtGui.QApplication.UnicodeUTF8))
        self.label_3.setText(QtGui.QApplication.translate("Form",
                                                          "<html><head/><body><p align=\"center\"><span style=\" font-weight:600;\">Method</span></p></body></html>",
                                                          None, QtGui.QApplication.UnicodeUTF8))
        self.save_cal.setText(QtGui.QApplication.translate("Form", "Save optimized calibration object", None,
                                                           QtGui.QApplication.UnicodeUTF8))
        self.label_7.setText(QtGui.QApplication.translate("Form",
                                                          "<html><head/><body><p align=\"right\"><span style=\" font-weight:600;\">Random generator seed:</span></p></body></html>",
                                                          None, QtGui.QApplication.UnicodeUTF8))
        self.bg_checkBox.setText(
            QtGui.QApplication.translate("Form", "Background compensation", None, QtGui.QApplication.UnicodeUTF8))
