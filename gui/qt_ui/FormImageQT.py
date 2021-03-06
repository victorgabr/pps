# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\vgalves\Dropbox\DFR\film2dose\qt_ui\FormImageUI.ui'
#
# Created: Thu May  5 15:22:46 2016
#      by: pyside-uic 0.2.15 running on PySide 1.2.2
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui


class Ui_imageForm(object):
    def setupUi(self, imageForm):
        imageForm.setObjectName("imageForm")
        imageForm.resize(811, 600)
        self.verticalLayout = QtGui.QVBoxLayout(imageForm)
        self.verticalLayout.setObjectName("verticalLayout")
        self.verticalLayout_2 = QtGui.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.gridLayout = QtGui.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.channel_box = QtGui.QComboBox(imageForm)
        self.channel_box.setObjectName("channel_box")
        self.channel_box.addItem("")
        self.channel_box.addItem("")
        self.channel_box.addItem("")
        self.gridLayout.addWidget(self.channel_box, 1, 0, 1, 1)
        self.button_flipud = QtGui.QPushButton(imageForm)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/icons/FUD.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.button_flipud.setIcon(icon)
        self.button_flipud.setObjectName("button_flipud")
        self.gridLayout.addWidget(self.button_flipud, 1, 8, 1, 1)
        self.save_as = QtGui.QPushButton(imageForm)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/icons/export.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.save_as.setIcon(icon1)
        self.save_as.setObjectName("save_as")
        self.gridLayout.addWidget(self.save_as, 1, 12, 1, 1)
        self.rotate_90cw = QtGui.QPushButton(imageForm)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/icons/rotate_cw.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.rotate_90cw.setIcon(icon2)
        self.rotate_90cw.setObjectName("rotate_90cw")
        self.gridLayout.addWidget(self.rotate_90cw, 0, 7, 1, 1)
        self.label = QtGui.QLabel(imageForm)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 1, 1, 1)
        self.rotate_90ccw = QtGui.QPushButton(imageForm)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/icons/rotate_ccw.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.rotate_90ccw.setIcon(icon3)
        self.rotate_90ccw.setObjectName("rotate_90ccw")
        self.gridLayout.addWidget(self.rotate_90ccw, 0, 8, 1, 1)
        self.isocenter_button = QtGui.QPushButton(imageForm)
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap(":/icons/target.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.isocenter_button.setIcon(icon4)
        self.isocenter_button.setIconSize(QtCore.QSize(16, 16))
        self.isocenter_button.setObjectName("isocenter_button")
        self.gridLayout.addWidget(self.isocenter_button, 0, 0, 1, 1)
        self.button_rotatePoints = QtGui.QPushButton(imageForm)
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap(":/icons/ROT_POINTS.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.button_rotatePoints.setIcon(icon5)
        self.button_rotatePoints.setObjectName("button_rotatePoints")
        self.gridLayout.addWidget(self.button_rotatePoints, 0, 12, 1, 1)
        self.button_fliplr = QtGui.QPushButton(imageForm)
        icon6 = QtGui.QIcon()
        icon6.addPixmap(QtGui.QPixmap(":/icons/FLR.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.button_fliplr.setIcon(icon6)
        self.button_fliplr.setObjectName("button_fliplr")
        self.gridLayout.addWidget(self.button_fliplr, 1, 7, 1, 1)
        self.label_3 = QtGui.QLabel(imageForm)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 0, 3, 1, 1)
        self.colorComboBox = QtGui.QComboBox(imageForm)
        self.colorComboBox.setObjectName("colorComboBox")
        self.gridLayout.addWidget(self.colorComboBox, 1, 1, 1, 1)
        self.label_2 = QtGui.QLabel(imageForm)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 0, 2, 1, 1)
        self.minLineEdit = QtGui.QLineEdit(imageForm)
        self.minLineEdit.setObjectName("minLineEdit")
        self.gridLayout.addWidget(self.minLineEdit, 1, 2, 1, 1)
        self.maxLineEdit = QtGui.QLineEdit(imageForm)
        self.maxLineEdit.setObjectName("maxLineEdit")
        self.gridLayout.addWidget(self.maxLineEdit, 1, 3, 1, 1)
        self.verticalLayout_2.addLayout(self.gridLayout)
        self.verticalLayout.addLayout(self.verticalLayout_2)

        self.retranslateUi(imageForm)
        QtCore.QMetaObject.connectSlotsByName(imageForm)

    def retranslateUi(self, imageForm):
        imageForm.setWindowTitle(
            QtGui.QApplication.translate("imageForm", "Form", None, QtGui.QApplication.UnicodeUTF8))
        self.channel_box.setToolTip(QtGui.QApplication.translate("imageForm",
                                                                 "<html><head/><body><p><span style=\" font-weight:600;\">Select image channel</span></p></body></html>",
                                                                 None, QtGui.QApplication.UnicodeUTF8))
        self.channel_box.setItemText(0, QtGui.QApplication.translate("imageForm", "Red channel", None,
                                                                     QtGui.QApplication.UnicodeUTF8))
        self.channel_box.setItemText(1, QtGui.QApplication.translate("imageForm", "Green channel", None,
                                                                     QtGui.QApplication.UnicodeUTF8))
        self.channel_box.setItemText(2, QtGui.QApplication.translate("imageForm", "Blue channel", None,
                                                                     QtGui.QApplication.UnicodeUTF8))
        self.button_flipud.setToolTip(
            QtGui.QApplication.translate("imageForm", "<html><head/><body><p>Vertical mirror.</p></body></html>", None,
                                         QtGui.QApplication.UnicodeUTF8))
        self.button_flipud.setText(
            QtGui.QApplication.translate("imageForm", "Flip UD", None, QtGui.QApplication.UnicodeUTF8))
        self.save_as.setToolTip(QtGui.QApplication.translate("imageForm",
                                                             "<html><head/><body><p>Save image as film2dose *.ftd</p></body></html>",
                                                             None, QtGui.QApplication.UnicodeUTF8))
        self.save_as.setText(QtGui.QApplication.translate("imageForm", "Save as", None, QtGui.QApplication.UnicodeUTF8))
        self.rotate_90cw.setToolTip(
            QtGui.QApplication.translate("imageForm", "<html><head/><body><p>rotate image clockwise</p></body></html>",
                                         None, QtGui.QApplication.UnicodeUTF8))
        self.rotate_90cw.setText(
            QtGui.QApplication.translate("imageForm", "Rotate CW", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("imageForm", "Colormap", None, QtGui.QApplication.UnicodeUTF8))
        self.rotate_90ccw.setToolTip(QtGui.QApplication.translate("imageForm",
                                                                  "<html><head/><body><p>rotate image counterclockwise</p></body></html>",
                                                                  None, QtGui.QApplication.UnicodeUTF8))
        self.rotate_90ccw.setText(
            QtGui.QApplication.translate("imageForm", "Rotate CCW", None, QtGui.QApplication.UnicodeUTF8))
        self.isocenter_button.setToolTip(QtGui.QApplication.translate("imageForm",
                                                                      "<html><head/><body><p>Set image coordinates isocenter</p></body></html>",
                                                                      None, QtGui.QApplication.UnicodeUTF8))
        self.isocenter_button.setText(
            QtGui.QApplication.translate("imageForm", "Set Iso", None, QtGui.QApplication.UnicodeUTF8))
        self.button_rotatePoints.setToolTip(QtGui.QApplication.translate("imageForm",
                                                                         "<html><head/><body><p><span style=\" font-weight:600;\">Rotate the image by selecting 2 reference points</span></p></body></html>",
                                                                         None, QtGui.QApplication.UnicodeUTF8))
        self.button_rotatePoints.setText(
            QtGui.QApplication.translate("imageForm", "Rotation points", None, QtGui.QApplication.UnicodeUTF8))
        self.button_fliplr.setToolTip(
            QtGui.QApplication.translate("imageForm", "<html><head/><body><p>Horizontal mirror.</p></body></html>",
                                         None, QtGui.QApplication.UnicodeUTF8))
        self.button_fliplr.setText(
            QtGui.QApplication.translate("imageForm", "Flip LR ", None, QtGui.QApplication.UnicodeUTF8))
        self.label_3.setText(
            QtGui.QApplication.translate("imageForm", "Window max.", None, QtGui.QApplication.UnicodeUTF8))
        self.colorComboBox.setToolTip(QtGui.QApplication.translate("imageForm",
                                                                   "<html><head/><body><p><span style=\" font-weight:600;\">Select image colormap</span></p></body></html>",
                                                                   None, QtGui.QApplication.UnicodeUTF8))
        self.label_2.setText(
            QtGui.QApplication.translate("imageForm", "Window min.", None, QtGui.QApplication.UnicodeUTF8))
