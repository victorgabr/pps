# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\vgalves\Dropbox\DFR\film2dose\qt_ui\dose_optim.ui'
#
# Created: Mon Dec 28 14:18:57 2015
#      by: pyside-uic 0.2.15 running on PySide 1.2.2
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui

class Ui_DoseOptimForm(object):
    def setupUi(self, DoseOptimForm):
        DoseOptimForm.setObjectName("DoseOptimForm")
        DoseOptimForm.resize(1078, 224)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/icons/App_icon.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        DoseOptimForm.setWindowIcon(icon)
        self.verticalLayout = QtGui.QVBoxLayout(DoseOptimForm)
        self.verticalLayout.setObjectName("verticalLayout")
        self.verticalLayout_2 = QtGui.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.gridLayout = QtGui.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.trim_button = QtGui.QPushButton(DoseOptimForm)
        self.trim_button.setObjectName("trim_button")
        self.gridLayout.addWidget(self.trim_button, 2, 4, 1, 1)
        self.radio_pixel = QtGui.QRadioButton(DoseOptimForm)
        self.radio_pixel.setObjectName("radio_pixel")
        self.gridLayout.addWidget(self.radio_pixel, 1, 3, 1, 1)
        self.button_rotatePoints = QtGui.QPushButton(DoseOptimForm)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/icons/ROT_POINTS.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.button_rotatePoints.setIcon(icon1)
        self.button_rotatePoints.setObjectName("button_rotatePoints")
        self.gridLayout.addWidget(self.button_rotatePoints, 1, 0, 1, 1)
        self.radio_mm = QtGui.QRadioButton(DoseOptimForm)
        self.radio_mm.setObjectName("radio_mm")
        self.gridLayout.addWidget(self.radio_mm, 1, 4, 1, 1)
        self.button_flipud = QtGui.QPushButton(DoseOptimForm)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/icons/FUD.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.button_flipud.setIcon(icon2)
        self.button_flipud.setObjectName("button_flipud")
        self.gridLayout.addWidget(self.button_flipud, 1, 1, 1, 1)
        self.button_fliplr = QtGui.QPushButton(DoseOptimForm)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/icons/FLR.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.button_fliplr.setIcon(icon3)
        self.button_fliplr.setObjectName("button_fliplr")
        self.gridLayout.addWidget(self.button_fliplr, 1, 2, 1, 1)
        self.rotate_90cw = QtGui.QPushButton(DoseOptimForm)
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap(":/icons/rotate_cw.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.rotate_90cw.setIcon(icon4)
        self.rotate_90cw.setObjectName("rotate_90cw")
        self.gridLayout.addWidget(self.rotate_90cw, 2, 1, 1, 1)
        self.rotate_90ccw = QtGui.QPushButton(DoseOptimForm)
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap(":/icons/rotate_ccw.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.rotate_90ccw.setIcon(icon5)
        self.rotate_90ccw.setObjectName("rotate_90ccw")
        self.gridLayout.addWidget(self.rotate_90ccw, 2, 2, 1, 1)
        self.label_3 = QtGui.QLabel(DoseOptimForm)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 2, 0, 1, 1)
        self.open_button = QtGui.QPushButton(DoseOptimForm)
        icon6 = QtGui.QIcon()
        icon6.addPixmap(QtGui.QPixmap(":/icons/Dosimetry.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.open_button.setIcon(icon6)
        self.open_button.setObjectName("open_button")
        self.gridLayout.addWidget(self.open_button, 4, 3, 1, 1)
        self.colorComboBox = QtGui.QComboBox(DoseOptimForm)
        self.colorComboBox.setObjectName("colorComboBox")
        self.gridLayout.addWidget(self.colorComboBox, 3, 0, 1, 1)
        self.channel_box = QtGui.QComboBox(DoseOptimForm)
        self.channel_box.setObjectName("channel_box")
        self.channel_box.addItem("")
        self.channel_box.addItem("")
        self.channel_box.addItem("")
        self.gridLayout.addWidget(self.channel_box, 4, 0, 1, 1)
        self.isocenter_button = QtGui.QPushButton(DoseOptimForm)
        icon7 = QtGui.QIcon()
        icon7.addPixmap(QtGui.QPixmap(":/icons/target.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.isocenter_button.setIcon(icon7)
        self.isocenter_button.setIconSize(QtCore.QSize(16, 16))
        self.isocenter_button.setObjectName("isocenter_button")
        self.gridLayout.addWidget(self.isocenter_button, 3, 3, 1, 1)
        self.save_as = QtGui.QPushButton(DoseOptimForm)
        icon8 = QtGui.QIcon()
        icon8.addPixmap(QtGui.QPixmap(":/icons/export.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.save_as.setIcon(icon8)
        self.save_as.setObjectName("save_as")
        self.gridLayout.addWidget(self.save_as, 4, 4, 1, 1)
        self.button_pointDose = QtGui.QPushButton(DoseOptimForm)
        icon9 = QtGui.QIcon()
        icon9.addPixmap(QtGui.QPixmap(":/icons/Point_dose.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.button_pointDose.setIcon(icon9)
        self.button_pointDose.setIconSize(QtCore.QSize(16, 16))
        self.button_pointDose.setObjectName("button_pointDose")
        self.gridLayout.addWidget(self.button_pointDose, 3, 4, 1, 1)
        self.multiply_button = QtGui.QPushButton(DoseOptimForm)
        self.multiply_button.setObjectName("multiply_button")
        self.gridLayout.addWidget(self.multiply_button, 2, 3, 1, 1)
        self.minLineEdit = QtGui.QLineEdit(DoseOptimForm)
        self.minLineEdit.setObjectName("minLineEdit")
        self.gridLayout.addWidget(self.minLineEdit, 4, 1, 1, 1)
        self.maxLineEdit = QtGui.QLineEdit(DoseOptimForm)
        self.maxLineEdit.setObjectName("maxLineEdit")
        self.gridLayout.addWidget(self.maxLineEdit, 4, 2, 1, 1)
        self.label = QtGui.QLabel(DoseOptimForm)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 3, 1, 1, 1)
        self.label_2 = QtGui.QLabel(DoseOptimForm)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 3, 2, 1, 1)
        self.verticalLayout_2.addLayout(self.gridLayout)
        self.verticalLayout.addLayout(self.verticalLayout_2)

        self.retranslateUi(DoseOptimForm)
        QtCore.QMetaObject.connectSlotsByName(DoseOptimForm)

    def retranslateUi(self, DoseOptimForm):
        DoseOptimForm.setWindowTitle(
            QtGui.QApplication.translate("DoseOptimForm", "Form", None, QtGui.QApplication.UnicodeUTF8))
        self.trim_button.setText(
            QtGui.QApplication.translate("DoseOptimForm", "Trim image", None, QtGui.QApplication.UnicodeUTF8))
        self.radio_pixel.setText(
            QtGui.QApplication.translate("DoseOptimForm", "scale in pixel", None, QtGui.QApplication.UnicodeUTF8))
        self.button_rotatePoints.setToolTip(QtGui.QApplication.translate("DoseOptimForm",
                                                                         "<html><head/><body><p><span style=\" font-weight:600;\">Rotate the image by selecting 2 reference points</span></p></body></html>",
                                                                         None, QtGui.QApplication.UnicodeUTF8))
        self.button_rotatePoints.setText(
            QtGui.QApplication.translate("DoseOptimForm", "Rotation points", None, QtGui.QApplication.UnicodeUTF8))
        self.radio_mm.setText(
            QtGui.QApplication.translate("DoseOptimForm", "scale in mm", None, QtGui.QApplication.UnicodeUTF8))
        self.button_flipud.setToolTip(
            QtGui.QApplication.translate("DoseOptimForm", "<html><head/><body><p>Vertical mirror.</p></body></html>",
                                         None, QtGui.QApplication.UnicodeUTF8))
        self.button_flipud.setText(
            QtGui.QApplication.translate("DoseOptimForm", "Flip UD", None, QtGui.QApplication.UnicodeUTF8))
        self.button_fliplr.setToolTip(
            QtGui.QApplication.translate("DoseOptimForm", "<html><head/><body><p>Horizontal mirror.</p></body></html>",
                                         None, QtGui.QApplication.UnicodeUTF8))
        self.button_fliplr.setText(
            QtGui.QApplication.translate("DoseOptimForm", "Flip LR ", None, QtGui.QApplication.UnicodeUTF8))
        self.rotate_90cw.setToolTip(QtGui.QApplication.translate("DoseOptimForm",
                                                                 "<html><head/><body><p>rotate image clockwise</p></body></html>",
                                                                 None, QtGui.QApplication.UnicodeUTF8))
        self.rotate_90cw.setText(
            QtGui.QApplication.translate("DoseOptimForm", "Rotate CW", None, QtGui.QApplication.UnicodeUTF8))
        self.rotate_90ccw.setToolTip(QtGui.QApplication.translate("DoseOptimForm",
                                                                  "<html><head/><body><p>rotate image counterclockwise</p></body></html>",
                                                                  None, QtGui.QApplication.UnicodeUTF8))
        self.rotate_90ccw.setText(
            QtGui.QApplication.translate("DoseOptimForm", "Rotate CCW", None, QtGui.QApplication.UnicodeUTF8))
        self.label_3.setText(QtGui.QApplication.translate("DoseOptimForm",
                                                          "<html><head/><body><p align=\"center\"><span style=\" font-weight:600;\">Colormap</span></p></body></html>",
                                                          None, QtGui.QApplication.UnicodeUTF8))
        self.open_button.setText(
            QtGui.QApplication.translate("DoseOptimForm", "Open file", None, QtGui.QApplication.UnicodeUTF8))
        self.colorComboBox.setToolTip(QtGui.QApplication.translate("DoseOptimForm",
                                                                   "<html><head/><body><p><span style=\" font-weight:600;\">Select image colormap</span></p></body></html>",
                                                                   None, QtGui.QApplication.UnicodeUTF8))
        self.channel_box.setToolTip(QtGui.QApplication.translate("DoseOptimForm",
                                                                 "<html><head/><body><p><span style=\" font-weight:600;\">Select image channel</span></p></body></html>",
                                                                 None, QtGui.QApplication.UnicodeUTF8))
        self.channel_box.setItemText(0, QtGui.QApplication.translate("DoseOptimForm", "Dose", None,
                                                                     QtGui.QApplication.UnicodeUTF8))
        self.channel_box.setItemText(1, QtGui.QApplication.translate("DoseOptimForm", "Disturbance ", None,
                                                                     QtGui.QApplication.UnicodeUTF8))
        self.channel_box.setItemText(2, QtGui.QApplication.translate("DoseOptimForm", "Type B uncertainty", None,
                                                                     QtGui.QApplication.UnicodeUTF8))
        self.isocenter_button.setToolTip(QtGui.QApplication.translate("DoseOptimForm",
                                                                      "<html><head/><body><p>Set image coordinates isocenter</p></body></html>",
                                                                      None, QtGui.QApplication.UnicodeUTF8))
        self.isocenter_button.setText(
            QtGui.QApplication.translate("DoseOptimForm", "Set Iso", None, QtGui.QApplication.UnicodeUTF8))
        self.save_as.setToolTip(QtGui.QApplication.translate("DoseOptimForm",
                                                             "<html><head/><body><p>Save image as film2dose *.ftd</p></body></html>",
                                                             None, QtGui.QApplication.UnicodeUTF8))
        self.save_as.setText(
            QtGui.QApplication.translate("DoseOptimForm", "Save as", None, QtGui.QApplication.UnicodeUTF8))
        self.button_pointDose.setToolTip(QtGui.QApplication.translate("DoseOptimForm",
                                                                      "<html><head/><body><p>Calculates the average dose and its uncertainty over a 10 mm x 10 mm area. </p></body></html>",
                                                                      None, QtGui.QApplication.UnicodeUTF8))
        self.button_pointDose.setText(
            QtGui.QApplication.translate("DoseOptimForm", "Point dose", None, QtGui.QApplication.UnicodeUTF8))
        self.multiply_button.setToolTip(QtGui.QApplication.translate("DoseOptimForm",
                                                                     "<html><head/><body><p>Multiplies the entire image matrix by a dimensionless factor</p></body></html>",
                                                                     None, QtGui.QApplication.UnicodeUTF8))
        self.multiply_button.setText(QtGui.QApplication.translate("DoseOptimForm", "Multiply by / Normalize", None,
                                                                  QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("DoseOptimForm",
                                                        "<html><head/><body><p align=\"center\"><span style=\" font-weight:600;\">Window min</span></p></body></html>",
                                                        None, QtGui.QApplication.UnicodeUTF8))
        self.label_2.setText(QtGui.QApplication.translate("DoseOptimForm",
                                                          "<html><head/><body><p align=\"center\"><span style=\" font-weight:600;\">Window max</span></p></body></html>",
                                                          None, QtGui.QApplication.UnicodeUTF8))


from film2dose.qt_ui import icons_rc
