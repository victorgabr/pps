# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import platform
import sys

import PySide
import matplotlib
from PySide import QtGui, QtCore

# from PySide.QtCore import QLocale

matplotlib.use('Qt4Agg')
matplotlib.rcParams['backend.qt4'] = 'PySide'

from film2dose.qt_ui import MainWindowQt
from film2dose.qt_ui.dockmanager import DockManager
from functools import partial
from film2dose.qt_ui.Film2DoseWidgets import CalibrationWidget, DoseConversionWidget, \
    BatchDoseConversionWidget, PicketFenceWidget, StarShotWidget, LicenceWidget, CalibrationModeWidget, \
    GetDataWidget, GammaComparisonWidget, DoseComparisonWidget, FieldWidgetTool

from film2dose.tools.misc import read_licence_key, check_licence

__version__ = '0.1.0'


def _sys_getenc_wrapper():
    return 'UTF-8'


sys.getfilesystemencoding = _sys_getenc_wrapper


# TODO Save picket Fence Report.

# noinspection PyUnresolvedReferences
class MainDialog(QtGui.QMainWindow, MainWindowQt.Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainDialog, self).__init__()
        self.cal = None
        self.im = None
        self.delta = None
        self.channel = 1  # red Channel
        self.figure_canvas = None
        self.navigation_toolbar = None
        self.calib_widget = None
        self.fit_widget = None
        self.tps_widget = None
        self.film2doseWidget = None
        self.plan_comp_widget = None
        self.gamma_comp_widget = None
        self.picket_fence_widget = None
        self.batchdoseWidget = None
        self.star_shot_widget = None
        self.field_widget = None
        self.setupUi(self)
        self.licence_widget = LicenceWidget()
        self.actionQuit.triggered.connect(self.exitApp)
        # start a dock manager on MainWidget
        self.dockmanager = DockManager(self, self.centralwidget)
        # Set connections
        self.set_conections()

    def set_conections(self):
        self.action48_bit_tiff_image.triggered.connect(partial(self.dockmanager.new_dock, 0))
        self.actionDose_Image.triggered.connect(partial(self.dockmanager.new_dock, 2))
        self.actionCommon_TPS_formats.triggered.connect(partial(self.dockmanager.new_dock, 1))
        self.actionFilm_Calibration.triggered.connect(self.load_cal_images)
        self.actionFitCurves.triggered.connect(self.load_calibration)
        self.actionDose_Conversion.triggered.connect(self.film2dose)
        self.actionPlan_Comparisson.triggered.connect(self.plan_comparison)
        self.actionGamma_Index_2.triggered.connect(self.on_gamma)

        # symmetry and flatness
        self.actionFlatness_and_Symmetry.triggered.connect(self.on_sym)
        # Picket fence
        self.actionPicket_Fence_2.triggered.connect(self.load_picketfence)
        # Batch Dose.
        self.actionBatch_film2dose.triggered.connect(self.on_batch)
        self.actionStarShot.triggered.connect(self.on_star_shot)

        # licence and About box
        self.actionLicence.triggered.connect(self.show_licence)
        self.actionAbout.triggered.connect(self.about)

    def on_sym(self):
        self.field_widget = FieldWidgetTool()
        self.field_widget.showMaximized()

    def on_gamma(self):
        self.plan_comp_widget = GammaComparisonWidget()
        self.plan_comp_widget.showMaximized()

    def plan_comparison(self):
        self.plan_comp_widget = DoseComparisonWidget()
        self.plan_comp_widget.showMaximized()

    def film2dose(self):
        self.film2doseWidget = DoseConversionWidget()
        self.film2doseWidget.showMaximized()

    def on_batch(self):
        self.batchdoseWidget = BatchDoseConversionWidget()
        self.batchdoseWidget.show()

    def load_calibration(self):
        self.fit_widget = CalibrationModeWidget()
        self.fit_widget.showMaximized()

    def load_cal_images(self):

        reply = QtGui.QMessageBox.question(None, "Warning",
                                           "Do you want to get calibration data using the point-based method?",
                                           QtGui.QMessageBox.Yes | QtGui.QMessageBox.No | QtGui.QMessageBox.Cancel)

        if reply == QtGui.QMessageBox.Yes:
            self.calib_widget = CalibrationWidget()
            self.calib_widget.show()
        elif reply == QtGui.QMessageBox.No:
            QtGui.QMessageBox.information(None, "Calibration Model",
                                          "Proceed to  the plane point-based method")

            self.calib_widget = GetDataWidget()
            self.calib_widget.show()
        else:
            self.questionLabel.setText("Cancel")

    def load_picketfence(self):
        self.picket_fence_widget = PicketFenceWidget()
        self.picket_fence_widget.showMaximized()

    def on_star_shot(self):
        self.star_shot_widget = StarShotWidget()
        self.star_shot_widget.showMaximized()

    def show_licence(self):
        self.licence_widget.showMaximized()

    def about(self):
        txt = "<b>Film2Dose %s </b>" \
              "<p> Copyright &copy; 2014-2015 Victor Gabriel, victorgabr@gmail.com " \
              "All rights reserved in accordance with BSD licence." \
              " <p>Platform details:<p> Python %s - PySide version %s - Qt version %s on %s" % (
                  __version__, platform.python_version(), PySide.__version__, PySide.QtCore.__version__,
                  platform.system())

        QtGui.QMessageBox.about(self, 'Information', txt)

    def exitApp(self):
        self.close()


def licence_check_launch(filename='licence.key'):
    app = QtGui.QApplication(sys.argv)
    form = MainDialog()

    key = read_licence_key(filename)
    if key != '':
        if check_licence(key):
            form.show()
            sys.exit(app.exec_())
        else:
            QtGui.QMessageBox.information(None,
                                          "Wrong key!", "You need to get your key!")

            app.quit()
    else:
        QtGui.QMessageBox.information(None,
                                      "Missing licence file!", "You need to get your key!")
        app.quit()


def free_launch():
    try:
        app = QtGui.QApplication(sys.argv)
    except RuntimeError:
        app = QtCore.QCoreApplication.instance(sys.argv)

    # app = QtGui.QApplication(sys.argv)

    current_locale = QtCore.QLocale()
    print(current_locale.name())
    form = MainDialog()
    form.showMaximized()
    sys.exit(app.exec_())
