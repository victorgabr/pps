# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\vgalves\Dropbox\DFR\film2dose\qt_ui\tps_widget.ui'
#
# Created: Mon Dec 28 11:47:40 2015
#      by: pyside-uic 0.2.15 running on PySide 1.2.2
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui


class Ui_imageForm(object):
    def setupUi(self, imageForm):
        imageForm.setObjectName("imageForm")
        imageForm.resize(778, 378)
        self.verticalLayout = QtGui.QVBoxLayout(imageForm)
        self.verticalLayout.setObjectName("verticalLayout")
        self.verticalLayout_2 = QtGui.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.gridLayout = QtGui.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.button_flipud = QtGui.QPushButton(imageForm)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/icons/FUD.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.button_flipud.setIcon(icon)
        self.button_flipud.setObjectName("button_flipud")
        self.gridLayout.addWidget(self.button_flipud, 3, 5, 1, 1)
        self.button_fliplr = QtGui.QPushButton(imageForm)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/icons/FLR.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.button_fliplr.setIcon(icon1)
        self.button_fliplr.setObjectName("button_fliplr")
        self.gridLayout.addWidget(self.button_fliplr, 3, 4, 1, 1)
        self.maxLineEdit = QtGui.QLineEdit(imageForm)
        self.maxLineEdit.setInputMask("")
        self.maxLineEdit.setObjectName("maxLineEdit")
        self.gridLayout.addWidget(self.maxLineEdit, 3, 3, 1, 1)
        self.colorComboBox = QtGui.QComboBox(imageForm)
        self.colorComboBox.setObjectName("colorComboBox")
        self.gridLayout.addWidget(self.colorComboBox, 3, 1, 1, 1)
        self.open_button = QtGui.QPushButton(imageForm)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/icons/Dosimetry.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.open_button.setIcon(icon2)
        self.open_button.setIconSize(QtCore.QSize(16, 16))
        self.open_button.setObjectName("open_button")
        self.gridLayout.addWidget(self.open_button, 3, 6, 1, 1)
        self.minLineEdit = QtGui.QLineEdit(imageForm)
        self.minLineEdit.setInputMask("")
        self.minLineEdit.setObjectName("minLineEdit")
        self.gridLayout.addWidget(self.minLineEdit, 3, 2, 1, 1)
        self.multiply_button = QtGui.QPushButton(imageForm)
        self.multiply_button.setObjectName("multiply_button")
        self.gridLayout.addWidget(self.multiply_button, 3, 7, 1, 1)
        self.label = QtGui.QLabel(imageForm)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 1, 1, 1, 1)
        self.label_2 = QtGui.QLabel(imageForm)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 1, 2, 1, 1)
        self.label_3 = QtGui.QLabel(imageForm)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 1, 3, 1, 1)
        self.rotate_90cw = QtGui.QPushButton(imageForm)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/icons/rotate_cw.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.rotate_90cw.setIcon(icon3)
        self.rotate_90cw.setObjectName("rotate_90cw")
        self.gridLayout.addWidget(self.rotate_90cw, 1, 4, 1, 1)
        self.rotate_90ccw = QtGui.QPushButton(imageForm)
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap(":/icons/rotate_ccw.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.rotate_90ccw.setIcon(icon4)
        self.rotate_90ccw.setObjectName("rotate_90ccw")
        self.gridLayout.addWidget(self.rotate_90ccw, 1, 5, 1, 1)
        self.radio_mm = QtGui.QRadioButton(imageForm)
        self.radio_mm.setObjectName("radio_mm")
        self.gridLayout.addWidget(self.radio_mm, 1, 6, 1, 1)
        self.radio_pixel = QtGui.QRadioButton(imageForm)
        self.radio_pixel.setObjectName("radio_pixel")
        self.gridLayout.addWidget(self.radio_pixel, 1, 7, 1, 1)
        self.verticalLayout_2.addLayout(self.gridLayout)
        self.verticalLayout.addLayout(self.verticalLayout_2)

        self.retranslateUi(imageForm)
        QtCore.QMetaObject.connectSlotsByName(imageForm)

    def retranslateUi(self, imageForm):
        imageForm.setWindowTitle(
            QtGui.QApplication.translate("imageForm", "Form", None, QtGui.QApplication.UnicodeUTF8))
        self.button_flipud.setToolTip(
            QtGui.QApplication.translate("imageForm", "<html><head/><body><p>Vertical mirror.</p></body></html>", None,
                                         QtGui.QApplication.UnicodeUTF8))
        self.button_flipud.setText(
            QtGui.QApplication.translate("imageForm", "Flip UD", None, QtGui.QApplication.UnicodeUTF8))
        self.button_fliplr.setToolTip(
            QtGui.QApplication.translate("imageForm", "<html><head/><body><p>Horizontal mirror.</p></body></html>",
                                         None, QtGui.QApplication.UnicodeUTF8))
        self.button_fliplr.setText(
            QtGui.QApplication.translate("imageForm", "Flip LR ", None, QtGui.QApplication.UnicodeUTF8))
        self.colorComboBox.setToolTip(QtGui.QApplication.translate("imageForm",
                                                                   "<html><head/><body><p><span style=\" font-weight:600;\">Select image colormap</span></p></body></html>",
                                                                   None, QtGui.QApplication.UnicodeUTF8))
        self.open_button.setToolTip(QtGui.QApplication.translate("imageForm",
                                                                 "<html><head/><body><p>Set image coordinates isocenter</p></body></html>",
                                                                 None, QtGui.QApplication.UnicodeUTF8))
        self.open_button.setText(
            QtGui.QApplication.translate("imageForm", "Open file", None, QtGui.QApplication.UnicodeUTF8))
        self.multiply_button.setText(
            QtGui.QApplication.translate("imageForm", "Normalize/Multiply by", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("imageForm",
                                                        "<html><head/><body><p align=\"center\"><span style=\" font-weight:600;\">Colormap</span></p></body></html>",
                                                        None, QtGui.QApplication.UnicodeUTF8))
        self.label_2.setText(QtGui.QApplication.translate("imageForm",
                                                          "<html><head/><body><p align=\"center\"><span style=\" font-weight:600;\">Window min</span></p></body></html>",
                                                          None, QtGui.QApplication.UnicodeUTF8))
        self.label_3.setText(QtGui.QApplication.translate("imageForm",
                                                          "<html><head/><body><p align=\"center\"><span style=\" font-weight:600;\">Window max</span></p></body></html>",
                                                          None, QtGui.QApplication.UnicodeUTF8))
        self.rotate_90cw.setToolTip(
            QtGui.QApplication.translate("imageForm", "<html><head/><body><p>rotate image clockwise</p></body></html>",
                                         None, QtGui.QApplication.UnicodeUTF8))
        self.rotate_90cw.setText(
            QtGui.QApplication.translate("imageForm", "Rotate CW", None, QtGui.QApplication.UnicodeUTF8))
        self.rotate_90ccw.setToolTip(QtGui.QApplication.translate("imageForm",
                                                                  "<html><head/><body><p>rotate image counterclockwise</p></body></html>",
                                                                  None, QtGui.QApplication.UnicodeUTF8))
        self.rotate_90ccw.setText(
            QtGui.QApplication.translate("imageForm", "Rotate CCW", None, QtGui.QApplication.UnicodeUTF8))
        self.radio_mm.setText(
            QtGui.QApplication.translate("imageForm", " scale in mm", None, QtGui.QApplication.UnicodeUTF8))
        self.radio_pixel.setText(
            QtGui.QApplication.translate("imageForm", "scale in pixel", None, QtGui.QApplication.UnicodeUTF8))
