# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\vgalves\Dropbox\DFR\film2dose\qt_ui\DoseComp.ui'
#
# Created: Tue Nov 17 13:25:25 2015
#      by: pyside-uic 0.2.15 running on PySide 1.2.2
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui

class Ui_DoseComp(object):
    def setupUi(self, DoseComp):
        DoseComp.setObjectName("DoseComp")
        DoseComp.resize(800, 595)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/icons/compare.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        DoseComp.setWindowIcon(icon)
        self.verticalLayout = QtGui.QVBoxLayout(DoseComp)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.formLayout = QtGui.QFormLayout()
        self.formLayout.setFieldGrowthPolicy(QtGui.QFormLayout.AllNonFixedFieldsGrow)
        self.formLayout.setObjectName("formLayout")
        self.dosePercentageLabel = QtGui.QLabel(DoseComp)
        self.dosePercentageLabel.setObjectName("dosePercentageLabel")
        self.formLayout.setWidget(0, QtGui.QFormLayout.LabelRole, self.dosePercentageLabel)
        self.dosePercentageDoubleSpinBox = QtGui.QDoubleSpinBox(DoseComp)
        self.dosePercentageDoubleSpinBox.setMaximum(20.0)
        self.dosePercentageDoubleSpinBox.setSingleStep(1.0)
        self.dosePercentageDoubleSpinBox.setProperty("value", 3.0)
        self.dosePercentageDoubleSpinBox.setObjectName("dosePercentageDoubleSpinBox")
        self.formLayout.setWidget(0, QtGui.QFormLayout.FieldRole, self.dosePercentageDoubleSpinBox)
        self.dTAMmLabel = QtGui.QLabel(DoseComp)
        self.dTAMmLabel.setObjectName("dTAMmLabel")
        self.formLayout.setWidget(1, QtGui.QFormLayout.LabelRole, self.dTAMmLabel)
        self.dTAMmSpinBox = QtGui.QSpinBox(DoseComp)
        self.dTAMmSpinBox.setMinimum(1)
        self.dTAMmSpinBox.setProperty("value", 3)
        self.dTAMmSpinBox.setObjectName("dTAMmSpinBox")
        self.formLayout.setWidget(1, QtGui.QFormLayout.FieldRole, self.dTAMmSpinBox)
        self.doseThresholdLabel = QtGui.QLabel(DoseComp)
        self.doseThresholdLabel.setObjectName("doseThresholdLabel")
        self.formLayout.setWidget(2, QtGui.QFormLayout.LabelRole, self.doseThresholdLabel)
        self.doseThresholdDoubleSpinBox = QtGui.QDoubleSpinBox(DoseComp)
        self.doseThresholdDoubleSpinBox.setSingleStep(5.0)
        self.doseThresholdDoubleSpinBox.setProperty("value", 10.0)
        self.doseThresholdDoubleSpinBox.setObjectName("doseThresholdDoubleSpinBox")
        self.formLayout.setWidget(2, QtGui.QFormLayout.FieldRole, self.doseThresholdDoubleSpinBox)
        self.horizontalLayout_2.addLayout(self.formLayout)
        self.local_checkBox = QtGui.QCheckBox(DoseComp)
        self.local_checkBox.setObjectName("local_checkBox")
        self.horizontalLayout_2.addWidget(self.local_checkBox)
        self.open_images = QtGui.QPushButton(DoseComp)
        self.open_images.setObjectName("open_images")
        self.horizontalLayout_2.addWidget(self.open_images)
        self.compare_button = QtGui.QPushButton(DoseComp)
        self.compare_button.setObjectName("compare_button")
        self.horizontalLayout_2.addWidget(self.compare_button)
        self.save_images = QtGui.QPushButton(DoseComp)
        self.save_images.setObjectName("save_images")
        self.horizontalLayout_2.addWidget(self.save_images)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.gridLayout = QtGui.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.film_widget = QtGui.QWidget(DoseComp)
        self.film_widget.setObjectName("film_widget")
        self.gridLayout.addWidget(self.film_widget, 0, 1, 1, 1)
        self.tps_widget = QtGui.QWidget(DoseComp)
        self.tps_widget.setObjectName("tps_widget")
        self.gridLayout.addWidget(self.tps_widget, 0, 0, 1, 1)
        self.gamma_widget = QtGui.QWidget(DoseComp)
        self.gamma_widget.setObjectName("gamma_widget")
        self.gridLayout.addWidget(self.gamma_widget, 1, 0, 1, 1)
        self.hist_widget = QtGui.QWidget(DoseComp)
        self.hist_widget.setObjectName("hist_widget")
        self.gridLayout.addWidget(self.hist_widget, 1, 1, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)

        self.retranslateUi(DoseComp)
        QtCore.QMetaObject.connectSlotsByName(DoseComp)

    def retranslateUi(self, DoseComp):
        DoseComp.setWindowTitle(QtGui.QApplication.translate("DoseComp", "Form", None, QtGui.QApplication.UnicodeUTF8))
        self.dosePercentageLabel.setText(
            QtGui.QApplication.translate("DoseComp", "Dose percentage (%)", None, QtGui.QApplication.UnicodeUTF8))
        self.dTAMmLabel.setText(
            QtGui.QApplication.translate("DoseComp", "DTA (mm)", None, QtGui.QApplication.UnicodeUTF8))
        self.doseThresholdLabel.setText(
            QtGui.QApplication.translate("DoseComp", "Dose threshold (%)", None, QtGui.QApplication.UnicodeUTF8))
        self.local_checkBox.setText(
            QtGui.QApplication.translate("DoseComp", "Local gamma index", None, QtGui.QApplication.UnicodeUTF8))
        self.open_images.setText(
            QtGui.QApplication.translate("DoseComp", "Open Images", None, QtGui.QApplication.UnicodeUTF8))
        self.compare_button.setText(
            QtGui.QApplication.translate("DoseComp", "Compare", None, QtGui.QApplication.UnicodeUTF8))
        self.save_images.setText(
            QtGui.QApplication.translate("DoseComp", "Save images", None, QtGui.QApplication.UnicodeUTF8))

from film2dose.qt_ui import icons_rc
