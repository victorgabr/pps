# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\Victor\Dropbox\DFR\film2dose\qt_ui\fit_curves_widget.ui'
#
# Created: Thu Oct  1 13:32:31 2015
#      by: pyside-uic 0.2.15 running on PySide 1.2.2
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1200, 523)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/icons/curvechart-edit.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        Form.setWindowIcon(icon)
        self.horizontalLayout = QtGui.QHBoxLayout(Form)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout_8 = QtGui.QVBoxLayout()
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.horizontalLayout_5 = QtGui.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.comboBox = QtGui.QComboBox(Form)
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.horizontalLayout_5.addWidget(self.comboBox)
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem)
        self.radioButton_eq1 = QtGui.QRadioButton(Form)
        self.radioButton_eq1.setObjectName("radioButton_eq1")
        self.horizontalLayout_5.addWidget(self.radioButton_eq1)
        self.RadioButton_eq2 = QtGui.QRadioButton(Form)
        self.RadioButton_eq2.setObjectName("RadioButton_eq2")
        self.horizontalLayout_5.addWidget(self.RadioButton_eq2)
        self.radioButton_3 = QtGui.QRadioButton(Form)
        self.radioButton_3.setObjectName("radioButton_3")
        self.horizontalLayout_5.addWidget(self.radioButton_3)
        self.select_curve = QtGui.QPushButton(Form)
        self.select_curve.setObjectName("select_curve")
        self.horizontalLayout_5.addWidget(self.select_curve)
        spacerItem1 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem1)
        self.finish_button = QtGui.QPushButton(Form)
        self.finish_button.setObjectName("finish_button")
        self.horizontalLayout_5.addWidget(self.finish_button)
        self.verticalLayout_8.addLayout(self.horizontalLayout_5)
        self.horizontalLayout.addLayout(self.verticalLayout_8)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(
            QtGui.QApplication.translate("Form", "Fit curve tool", None, QtGui.QApplication.UnicodeUTF8))
        self.comboBox.setItemText(0, QtGui.QApplication.translate("Form", "Red Channel", None,
                                                                  QtGui.QApplication.UnicodeUTF8))
        self.comboBox.setItemText(1, QtGui.QApplication.translate("Form", "Green Channel", None,
                                                                  QtGui.QApplication.UnicodeUTF8))
        self.comboBox.setItemText(2, QtGui.QApplication.translate("Form", "Blue Channel", None,
                                                                  QtGui.QApplication.UnicodeUTF8))
        self.radioButton_eq1.setText(
            QtGui.QApplication.translate("Form", "Equation 1", None, QtGui.QApplication.UnicodeUTF8))
        self.RadioButton_eq2.setText(
            QtGui.QApplication.translate("Form", "Equation 2", None, QtGui.QApplication.UnicodeUTF8))
        self.radioButton_3.setText(
            QtGui.QApplication.translate("Form", "Equation 3", None, QtGui.QApplication.UnicodeUTF8))
        self.select_curve.setText(QtGui.QApplication.translate("Form", "Select", None, QtGui.QApplication.UnicodeUTF8))
        self.finish_button.setText(QtGui.QApplication.translate("Form", "Finish", None, QtGui.QApplication.UnicodeUTF8))


from film2dose.qt_ui import icons_rc
