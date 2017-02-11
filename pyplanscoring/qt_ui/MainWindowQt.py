# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '/home/victor/Dropbox/DFR/film2dose/qt_ui/Film2doseMainWindow.ui'
#
# Created: Sat Jun 25 12:16:37 2016
#      by: pyside-uic 0.2.15 running on PySide 1.2.4
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1436, 865)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/icons/App_icon.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtGui.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1436, 25))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtGui.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuNew_Project = QtGui.QMenu(self.menuFile)
        self.menuNew_Project.setObjectName("menuNew_Project")
        self.menuExport = QtGui.QMenu(self.menuFile)
        self.menuExport.setObjectName("menuExport")
        self.menuImport = QtGui.QMenu(self.menuFile)
        self.menuImport.setObjectName("menuImport")
        self.menuEdit = QtGui.QMenu(self.menubar)
        self.menuEdit.setObjectName("menuEdit")
        self.menuAnalysis = QtGui.QMenu(self.menubar)
        self.menuAnalysis.setObjectName("menuAnalysis")
        self.menuQA = QtGui.QMenu(self.menuAnalysis)
        self.menuQA.setObjectName("menuQA")
        self.menuAbout = QtGui.QMenu(self.menubar)
        self.menuAbout.setObjectName("menuAbout")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.toolBar = QtGui.QToolBar(MainWindow)
        self.toolBar.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.toolBar.setObjectName("toolBar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)
        self.actionFilm_Calibration = QtGui.QAction(MainWindow)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/icons/scanner.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionFilm_Calibration.setIcon(icon1)
        self.actionFilm_Calibration.setObjectName("actionFilm_Calibration")
        self.actionPlan_Comparisson = QtGui.QAction(MainWindow)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/icons/compare.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionPlan_Comparisson.setIcon(icon2)
        self.actionPlan_Comparisson.setObjectName("actionPlan_Comparisson")
        self.actionDICOM_RT = QtGui.QAction(MainWindow)
        self.actionDICOM_RT.setObjectName("actionDICOM_RT")
        self.actionClose = QtGui.QAction(MainWindow)
        self.actionClose.setObjectName("actionClose")
        self.actionQuit = QtGui.QAction(MainWindow)
        self.actionQuit.setObjectName("actionQuit")
        self.action48_bit_tiff_image = QtGui.QAction(MainWindow)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/icons/Import Picture Document.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action48_bit_tiff_image.setIcon(icon3)
        self.action48_bit_tiff_image.setObjectName("action48_bit_tiff_image")
        self.actionCommon_TPS_formats = QtGui.QAction(MainWindow)
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap(":/icons/import1.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionCommon_TPS_formats.setIcon(icon4)
        self.actionCommon_TPS_formats.setObjectName("actionCommon_TPS_formats")
        self.actionFitCurves = QtGui.QAction(MainWindow)
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap(":/icons/curvechart-edit.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionFitCurves.setIcon(icon5)
        self.actionFitCurves.setObjectName("actionFitCurves")
        self.actionFilm2dose = QtGui.QAction(MainWindow)
        self.actionFilm2dose.setObjectName("actionFilm2dose")
        self.actionGamma_Index = QtGui.QAction(MainWindow)
        self.actionGamma_Index.setObjectName("actionGamma_Index")
        self.actionProfiles = QtGui.QAction(MainWindow)
        self.actionProfiles.setObjectName("actionProfiles")
        self.actionStartshot = QtGui.QAction(MainWindow)
        self.actionStartshot.setObjectName("actionStartshot")
        self.actionPicket_Fence = QtGui.QAction(MainWindow)
        self.actionPicket_Fence.setObjectName("actionPicket_Fence")
        self.actionRestore_Image = QtGui.QAction(MainWindow)
        self.actionRestore_Image.setObjectName("actionRestore_Image")
        self.action90_degrees_CW = QtGui.QAction(MainWindow)
        icon6 = QtGui.QIcon()
        icon6.addPixmap(QtGui.QPixmap(":/icons/rotate_cw.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action90_degrees_CW.setIcon(icon6)
        self.action90_degrees_CW.setObjectName("action90_degrees_CW")
        self.action90_degrees_CCW = QtGui.QAction(MainWindow)
        icon7 = QtGui.QIcon()
        icon7.addPixmap(QtGui.QPixmap(":/icons/rotate_ccw.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action90_degrees_CCW.setIcon(icon7)
        self.action90_degrees_CCW.setObjectName("action90_degrees_CCW")
        self.actionGrid = QtGui.QAction(MainWindow)
        self.actionGrid.setObjectName("actionGrid")
        self.actionColormap = QtGui.QAction(MainWindow)
        self.actionColormap.setObjectName("actionColormap")
        self.actionMean_Value = QtGui.QAction(MainWindow)
        self.actionMean_Value.setObjectName("actionMean_Value")
        self.actionPicket_Fence_2 = QtGui.QAction(MainWindow)
        icon8 = QtGui.QIcon()
        icon8.addPixmap(QtGui.QPixmap(":/icons/grid.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionPicket_Fence_2.setIcon(icon8)
        self.actionPicket_Fence_2.setObjectName("actionPicket_Fence_2")
        self.actionStarShot = QtGui.QAction(MainWindow)
        self.actionStarShot.setObjectName("actionStarShot")
        self.actionFlatness_and_Symmetry = QtGui.QAction(MainWindow)
        self.actionFlatness_and_Symmetry.setObjectName("actionFlatness_and_Symmetry")
        self.actionDose_Conversion = QtGui.QAction(MainWindow)
        self.actionDose_Conversion.setIcon(icon)
        self.actionDose_Conversion.setObjectName("actionDose_Conversion")
        self.actionROI = QtGui.QAction(MainWindow)
        self.actionROI.setObjectName("actionROI")
        self.actionHorizontal_Flip = QtGui.QAction(MainWindow)
        self.actionHorizontal_Flip.setObjectName("actionHorizontal_Flip")
        self.actionVertical_Flip = QtGui.QAction(MainWindow)
        self.actionVertical_Flip.setObjectName("actionVertical_Flip")
        self.actionBatch_film2dose = QtGui.QAction(MainWindow)
        self.actionBatch_film2dose.setIcon(icon)
        self.actionBatch_film2dose.setObjectName("actionBatch_film2dose")
        self.actionLicence = QtGui.QAction(MainWindow)
        self.actionLicence.setObjectName("actionLicence")
        self.actionAbout = QtGui.QAction(MainWindow)
        self.actionAbout.setObjectName("actionAbout")
        self.actionDose_Image = QtGui.QAction(MainWindow)
        icon9 = QtGui.QIcon()
        icon9.addPixmap(QtGui.QPixmap(":/icons/Dosimetry.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionDose_Image.setIcon(icon9)
        self.actionDose_Image.setObjectName("actionDose_Image")
        self.actionS = QtGui.QAction(MainWindow)
        self.actionS.setObjectName("actionS")
        self.actionS_2 = QtGui.QAction(MainWindow)
        self.actionS_2.setObjectName("actionS_2")
        self.actionGamma_Index_2 = QtGui.QAction(MainWindow)
        icon10 = QtGui.QIcon()
        icon10.addPixmap(QtGui.QPixmap(":/icons/Greek_Gamma.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionGamma_Index_2.setIcon(icon10)
        self.actionGamma_Index_2.setObjectName("actionGamma_Index_2")
        self.menuNew_Project.addAction(self.actionFilm_Calibration)
        self.menuNew_Project.addAction(self.actionPlan_Comparisson)
        self.menuNew_Project.addAction(self.actionDose_Conversion)
        self.menuExport.addAction(self.actionDICOM_RT)
        self.menuExport.addAction(self.actionFilm2dose)
        self.menuImport.addAction(self.action48_bit_tiff_image)
        self.menuImport.addAction(self.actionCommon_TPS_formats)
        self.menuFile.addAction(self.menuImport.menuAction())
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.menuNew_Project.menuAction())
        self.menuFile.addAction(self.menuExport.menuAction())
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionClose)
        self.menuFile.addAction(self.actionQuit)
        self.menuEdit.addSeparator()
        self.menuQA.addAction(self.actionFlatness_and_Symmetry)
        self.menuQA.addAction(self.actionStarShot)
        self.menuAnalysis.addSeparator()
        self.menuAnalysis.addAction(self.menuQA.menuAction())
        self.menuAbout.addAction(self.actionLicence)
        self.menuAbout.addAction(self.actionAbout)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuEdit.menuAction())
        self.menubar.addAction(self.menuAnalysis.menuAction())
        self.menubar.addAction(self.menuAbout.menuAction())
        self.toolBar.addAction(self.action48_bit_tiff_image)
        self.toolBar.addAction(self.actionDose_Image)
        self.toolBar.addAction(self.actionCommon_TPS_formats)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actionFilm_Calibration)
        self.toolBar.addAction(self.actionFitCurves)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actionDose_Conversion)
        self.toolBar.addAction(self.actionBatch_film2dose)
        self.toolBar.addAction(self.actionGamma_Index_2)
        self.toolBar.addAction(self.actionPlan_Comparisson)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actionPicket_Fence_2)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(
            QtGui.QApplication.translate("MainWindow", "Film2dose", None, QtGui.QApplication.UnicodeUTF8))
        self.menuFile.setTitle(QtGui.QApplication.translate("MainWindow", "File", None, QtGui.QApplication.UnicodeUTF8))
        self.menuNew_Project.setTitle(
            QtGui.QApplication.translate("MainWindow", "New Project", None, QtGui.QApplication.UnicodeUTF8))
        self.menuExport.setTitle(
            QtGui.QApplication.translate("MainWindow", "Export ", None, QtGui.QApplication.UnicodeUTF8))
        self.menuImport.setTitle(
            QtGui.QApplication.translate("MainWindow", "Import Image", None, QtGui.QApplication.UnicodeUTF8))
        self.menuEdit.setTitle(QtGui.QApplication.translate("MainWindow", "Edit", None, QtGui.QApplication.UnicodeUTF8))
        self.menuAnalysis.setTitle(
            QtGui.QApplication.translate("MainWindow", "Analysis", None, QtGui.QApplication.UnicodeUTF8))
        self.menuQA.setTitle(QtGui.QApplication.translate("MainWindow", "QA", None, QtGui.QApplication.UnicodeUTF8))
        self.menuAbout.setTitle(
            QtGui.QApplication.translate("MainWindow", "About", None, QtGui.QApplication.UnicodeUTF8))
        self.toolBar.setWindowTitle(
            QtGui.QApplication.translate("MainWindow", "toolBar", None, QtGui.QApplication.UnicodeUTF8))
        self.actionFilm_Calibration.setText(
            QtGui.QApplication.translate("MainWindow", "Film Calibration", None, QtGui.QApplication.UnicodeUTF8))
        self.actionPlan_Comparisson.setText(
            QtGui.QApplication.translate("MainWindow", "Plan Comparisson", None, QtGui.QApplication.UnicodeUTF8))
        self.actionPlan_Comparisson.setIconText(
            QtGui.QApplication.translate("MainWindow", "Dose Comparison", None, QtGui.QApplication.UnicodeUTF8))
        self.actionDICOM_RT.setText(
            QtGui.QApplication.translate("MainWindow", "DICOM RT", None, QtGui.QApplication.UnicodeUTF8))
        self.actionClose.setText(
            QtGui.QApplication.translate("MainWindow", "Close ", None, QtGui.QApplication.UnicodeUTF8))
        self.actionQuit.setText(
            QtGui.QApplication.translate("MainWindow", "Quit", None, QtGui.QApplication.UnicodeUTF8))
        self.action48_bit_tiff_image.setText(
            QtGui.QApplication.translate("MainWindow", "Import Image", None, QtGui.QApplication.UnicodeUTF8))
        self.action48_bit_tiff_image.setIconText(
            QtGui.QApplication.translate("MainWindow", "Import Tiff", None, QtGui.QApplication.UnicodeUTF8))
        self.actionCommon_TPS_formats.setText(
            QtGui.QApplication.translate("MainWindow", "TPS formats", None, QtGui.QApplication.UnicodeUTF8))
        self.actionCommon_TPS_formats.setIconText(
            QtGui.QApplication.translate("MainWindow", "TPS", None, QtGui.QApplication.UnicodeUTF8))
        self.actionFitCurves.setText(
            QtGui.QApplication.translate("MainWindow", "Curve Fit", None, QtGui.QApplication.UnicodeUTF8))
        self.actionFilm2dose.setText(
            QtGui.QApplication.translate("MainWindow", "Film2dose ", None, QtGui.QApplication.UnicodeUTF8))
        self.actionGamma_Index.setText(
            QtGui.QApplication.translate("MainWindow", "Gamma Index", None, QtGui.QApplication.UnicodeUTF8))
        self.actionProfiles.setText(
            QtGui.QApplication.translate("MainWindow", "Profiles", None, QtGui.QApplication.UnicodeUTF8))
        self.actionStartshot.setText(
            QtGui.QApplication.translate("MainWindow", "Startshot", None, QtGui.QApplication.UnicodeUTF8))
        self.actionPicket_Fence.setText(
            QtGui.QApplication.translate("MainWindow", "Picket Fence", None, QtGui.QApplication.UnicodeUTF8))
        self.actionRestore_Image.setText(
            QtGui.QApplication.translate("MainWindow", "Restore image", None, QtGui.QApplication.UnicodeUTF8))
        self.action90_degrees_CW.setText(
            QtGui.QApplication.translate("MainWindow", "90 degrees CW", None, QtGui.QApplication.UnicodeUTF8))
        self.action90_degrees_CCW.setText(
            QtGui.QApplication.translate("MainWindow", "90 degrees CCW", None, QtGui.QApplication.UnicodeUTF8))
        self.actionGrid.setText(
            QtGui.QApplication.translate("MainWindow", "Grid", None, QtGui.QApplication.UnicodeUTF8))
        self.actionColormap.setText(
            QtGui.QApplication.translate("MainWindow", "Colormap", None, QtGui.QApplication.UnicodeUTF8))
        self.actionMean_Value.setText(
            QtGui.QApplication.translate("MainWindow", "Mean Value", None, QtGui.QApplication.UnicodeUTF8))
        self.actionPicket_Fence_2.setText(
            QtGui.QApplication.translate("MainWindow", "Picket Fence", None, QtGui.QApplication.UnicodeUTF8))
        self.actionStarShot.setText(
            QtGui.QApplication.translate("MainWindow", "StarShot", None, QtGui.QApplication.UnicodeUTF8))
        self.actionFlatness_and_Symmetry.setText(
            QtGui.QApplication.translate("MainWindow", "Flatness and Symmetry", None, QtGui.QApplication.UnicodeUTF8))
        self.actionDose_Conversion.setText(
            QtGui.QApplication.translate("MainWindow", "Dose Conversion", None, QtGui.QApplication.UnicodeUTF8))
        self.actionROI.setText(QtGui.QApplication.translate("MainWindow", "ROI", None, QtGui.QApplication.UnicodeUTF8))
        self.actionHorizontal_Flip.setText(
            QtGui.QApplication.translate("MainWindow", "Horizontal Flip", None, QtGui.QApplication.UnicodeUTF8))
        self.actionVertical_Flip.setText(
            QtGui.QApplication.translate("MainWindow", "Vertical Flip", None, QtGui.QApplication.UnicodeUTF8))
        self.actionBatch_film2dose.setText(
            QtGui.QApplication.translate("MainWindow", "Batch film2dose", None, QtGui.QApplication.UnicodeUTF8))
        self.actionBatch_film2dose.setIconText(
            QtGui.QApplication.translate("MainWindow", "Batch Dose Conversion", None, QtGui.QApplication.UnicodeUTF8))
        self.actionLicence.setText(
            QtGui.QApplication.translate("MainWindow", "Licence", None, QtGui.QApplication.UnicodeUTF8))
        self.actionAbout.setText(
            QtGui.QApplication.translate("MainWindow", "About", None, QtGui.QApplication.UnicodeUTF8))
        self.actionDose_Image.setText(
            QtGui.QApplication.translate("MainWindow", "Dose Image", None, QtGui.QApplication.UnicodeUTF8))
        self.actionS.setText(QtGui.QApplication.translate("MainWindow", "s", None, QtGui.QApplication.UnicodeUTF8))
        self.actionS_2.setText(QtGui.QApplication.translate("MainWindow", "s", None, QtGui.QApplication.UnicodeUTF8))
        self.actionGamma_Index_2.setText(
            QtGui.QApplication.translate("MainWindow", "Gamma Index", None, QtGui.QApplication.UnicodeUTF8))
