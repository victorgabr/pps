# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '/home/victor/Dropbox/DFR/film2dose/qt_ui/PicketFence.ui'
#
# Created: Sat Jun 25 12:17:33 2016
#      by: pyside-uic 0.2.15 running on PySide 1.2.4
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui


class Ui_DoseComp(object):
    def setupUi(self, DoseComp):
        DoseComp.setObjectName("DoseComp")
        DoseComp.resize(1055, 578)
        self.verticalLayout = QtGui.QVBoxLayout(DoseComp)
        self.verticalLayout.setObjectName("verticalLayout")
        self.vert_layout = QtGui.QVBoxLayout()
        self.vert_layout.setObjectName("vert_layout")
        self.gridLayout = QtGui.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.label_2 = QtGui.QLabel(DoseComp)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 0, 1, 1, 1)
        self.label_3 = QtGui.QLabel(DoseComp)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 0, 2, 1, 1)
        self.label = QtGui.QLabel(DoseComp)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.strsize_spinBox = QtGui.QDoubleSpinBox(DoseComp)
        self.strsize_spinBox.setObjectName("strsize_spinBox")
        self.gridLayout.addWidget(self.strsize_spinBox, 2, 2, 1, 1)
        self.separation_spinBox = QtGui.QDoubleSpinBox(DoseComp)
        self.separation_spinBox.setObjectName("separation_spinBox")
        self.gridLayout.addWidget(self.separation_spinBox, 2, 3, 1, 1)
        self.label_5 = QtGui.QLabel(DoseComp)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 0, 4, 1, 1)
        self.nstrips_spinBox = QtGui.QSpinBox(DoseComp)
        self.nstrips_spinBox.setObjectName("nstrips_spinBox")
        self.gridLayout.addWidget(self.nstrips_spinBox, 2, 1, 1, 1)
        self.label_4 = QtGui.QLabel(DoseComp)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 0, 3, 1, 1)
        self.comboBox = QtGui.QComboBox(DoseComp)
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.gridLayout.addWidget(self.comboBox, 2, 0, 1, 1)
        self.tolerance_SpinBox = QtGui.QDoubleSpinBox(DoseComp)
        self.tolerance_SpinBox.setObjectName("tolerance_SpinBox")
        self.gridLayout.addWidget(self.tolerance_SpinBox, 2, 4, 1, 1)
        self.analyse_button = QtGui.QPushButton(DoseComp)
        self.analyse_button.setObjectName("analyse_button")
        self.gridLayout.addWidget(self.analyse_button, 2, 5, 1, 1)
        self.label_6 = QtGui.QLabel(DoseComp)
        self.label_6.setObjectName("label_6")
        self.gridLayout.addWidget(self.label_6, 0, 5, 1, 1)
        self.save_results_button = QtGui.QPushButton(DoseComp)
        self.save_results_button.setObjectName("save_results_button")
        self.gridLayout.addWidget(self.save_results_button, 3, 5, 1, 1)
        self.trim_x_spin = QtGui.QDoubleSpinBox(DoseComp)
        self.trim_x_spin.setObjectName("trim_x_spin")
        self.gridLayout.addWidget(self.trim_x_spin, 3, 3, 1, 1)
        self.trim_y_spin = QtGui.QDoubleSpinBox(DoseComp)
        self.trim_y_spin.setObjectName("trim_y_spin")
        self.gridLayout.addWidget(self.trim_y_spin, 3, 4, 1, 1)
        self.restore_button = QtGui.QPushButton(DoseComp)
        self.restore_button.setObjectName("restore_button")
        self.gridLayout.addWidget(self.restore_button, 3, 0, 1, 1)
        self.trim_button = QtGui.QPushButton(DoseComp)
        self.trim_button.setObjectName("trim_button")
        self.gridLayout.addWidget(self.trim_button, 3, 2, 1, 1)
        self.thresh_button = QtGui.QPushButton(DoseComp)
        self.thresh_button.setObjectName("thresh_button")
        self.gridLayout.addWidget(self.thresh_button, 3, 1, 1, 1)
        self.vert_layout.addLayout(self.gridLayout)
        self.verticalLayout.addLayout(self.vert_layout)

        self.retranslateUi(DoseComp)
        QtCore.QMetaObject.connectSlotsByName(DoseComp)

    def retranslateUi(self, DoseComp):
        DoseComp.setWindowTitle(
            QtGui.QApplication.translate("DoseComp", "Picket Fence Analysis", None, QtGui.QApplication.UnicodeUTF8))
        self.label_2.setText(
            QtGui.QApplication.translate("DoseComp", "Number of Strips", None, QtGui.QApplication.UnicodeUTF8))
        self.label_3.setText(
            QtGui.QApplication.translate("DoseComp", "Strips Sizes (mm)", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("DoseComp", "MLC Model", None, QtGui.QApplication.UnicodeUTF8))
        self.label_5.setText(
            QtGui.QApplication.translate("DoseComp", "Tolerance (mm)", None, QtGui.QApplication.UnicodeUTF8))
        self.label_4.setText(
            QtGui.QApplication.translate("DoseComp", "Separation (mm)", None, QtGui.QApplication.UnicodeUTF8))
        self.comboBox.setItemText(0, QtGui.QApplication.translate("DoseComp", "Millenium 120", None,
                                                                  QtGui.QApplication.UnicodeUTF8))
        self.comboBox.setItemText(1, QtGui.QApplication.translate("DoseComp", "HD 120", None,
                                                                  QtGui.QApplication.UnicodeUTF8))
        self.comboBox.setItemText(2, QtGui.QApplication.translate("DoseComp", "MLCi", None,
                                                                  QtGui.QApplication.UnicodeUTF8))
        self.comboBox.setItemText(3, QtGui.QApplication.translate("DoseComp", "Beam Modulator mMLC", None,
                                                                  QtGui.QApplication.UnicodeUTF8))
        self.analyse_button.setText(
            QtGui.QApplication.translate("DoseComp", "Analyse", None, QtGui.QApplication.UnicodeUTF8))
        self.label_6.setText(QtGui.QApplication.translate("DoseComp", "Results", None, QtGui.QApplication.UnicodeUTF8))
        self.save_results_button.setText(
            QtGui.QApplication.translate("DoseComp", "Save Results", None, QtGui.QApplication.UnicodeUTF8))
        self.restore_button.setText(
            QtGui.QApplication.translate("DoseComp", "Restore Image", None, QtGui.QApplication.UnicodeUTF8))
        self.trim_button.setText(
            QtGui.QApplication.translate("DoseComp", "Trim Image xy (mm)", None, QtGui.QApplication.UnicodeUTF8))
        self.thresh_button.setText(
            QtGui.QApplication.translate("DoseComp", "Image Thresholding", None, QtGui.QApplication.UnicodeUTF8))
