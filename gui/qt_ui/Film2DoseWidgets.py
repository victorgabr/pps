# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import matplotlib.pyplot as plt
import numpy as np
from PySide import QtCore
from PySide import QtGui
from film2dose.pyplanscoring.core.calibration import save, load, save_ftd, Film2DoseCalibration, Model
from film2dose.pyplanscoring.core.image import read_tiff, display_fig, read_dicom, get_crop, Fim2DoseImage, ImageRegistration, \
    rotate_image, read_brainlab, save_dicom_dose, image_crop, read_monaco, read_cal_doses, read_OmniPro, \
    Film2DoseBlockingMouseInput, pixel2od, plot_cal_data, dowsampling_image, od2pixel, dose_difference, \
    SymmetryFlatness, FusedImages, ProfileComparison, OctaviusFiles, image_trim_xy, auto_threshold
from film2dose.pyplanscoring.core.libmath import gamma_index, get_covarmatrix, analyse_roi, meanval_uncertainty
from film2dose.pyplanscoring.core.picketfence import VarianMLC, ElektaMLCi2, ElektaBeamModulatorMLC, PicketFenceSettings, \
    PicketFenceTest
from film2dose.pyplanscoring.core.starshot import StarShot
from film2dose.qt_ui import PicketFenceQT, DoseConversionQT, DoseCompQT, FormImageQT, FitCurvesQt, StarShotQT, \
    DoseOptimizedQT, TPSWidgetQT, FitModeQT, GetCalPointsQT, EditGridQT, OptimizationQT, DoseCompAbsQT, FieldWidgetQT
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar2
from matplotlib.blocking_input import BlockingMouseInput
from matplotlib.widgets import Cursor


# TODO add image header imformation on image widgets

class PicketFenceWidget(QtGui.QWidget, PicketFenceQT.Ui_DoseComp):
    def __init__(self, parent=None):
        super(PicketFenceWidget, self).__init__(parent)
        self.setupUi(self)
        self.image_widget = EditImageWidget()
        self.image_widget.read_image()
        self.image_widget.set_colormap('Greys')
        self.image_widget.show_image()
        self.vert_layout.addWidget(self.image_widget)
        self.mlc_model = {0: VarianMLC(), 1: VarianMLC(hd120=True), 2: ElektaMLCi2(), 3: ElektaBeamModulatorMLC()}
        self.ax = None
        self.fig_result = None
        self.result_widget = PicketFenceResultWidget()
        self.set_connections()
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Policy(5), QtGui.QSizePolicy.Policy(5))
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        self.image_widget.setSizePolicy(sizePolicy)
        # default analysis variables
        self.n_strips = 5
        self.nstrips_spinBox.setValue(self.n_strips)
        self.strip_size = 5.0  # mm
        self.strsize_spinBox.setValue(self.strip_size)
        self.separation = 30.0  # mm
        self.separation_spinBox.setValue(self.separation)
        self.tolerance = 0.5  # mm
        self.tolerance_SpinBox.setValue(self.tolerance)
        self.x_trim = 10  # mm
        self.trim_x_spin.setValue(self.x_trim)
        self.y_trim = 10  # mm
        self.trim_y_spin.setValue(self.y_trim)
        self.calculated = False
        self.picket_fence_test = None
        self.result_data = {}

    def set_connections(self):
        self.analyse_button.clicked.connect(self.analyse)
        self.nstrips_spinBox.valueChanged[str].connect(self.on_n_strips)
        self.strsize_spinBox.valueChanged[str].connect(self.on_strip_size)
        self.separation_spinBox.valueChanged[str].connect(self.on_separation)
        self.tolerance_SpinBox.valueChanged[str].connect(self.on_tolerance)
        self.trim_x_spin.valueChanged[str].connect(self.on_x_trim)
        self.trim_y_spin.valueChanged[str].connect(self.on_y_trim)
        self.save_results_button.clicked.connect(self.on_save)
        self.restore_button.clicked.connect(self.on_restore)
        self.trim_button.clicked.connect(self.on_trim)
        self.thresh_button.clicked.connect(self.on_thresh)

    def on_thresh(self):
        self.image_widget.image_threshold()

    def on_x_trim(self, val):
        self.x_trim = float(val.replace(',', '.'))

    def on_y_trim(self, val):
        self.y_trim = float(val.replace(',', '.'))

    def on_trim(self):
        self.image_widget.image_trim(self.x_trim, self.y_trim)

    def on_restore(self):
        self.image_widget.restore_image()

    def on_n_strips(self, val):
        self.n_strips = int(val)

    def on_strip_size(self, val):
        self.strip_size = float(val.replace(',', '.'))

    def on_separation(self, val):
        self.separation = float(val.replace(',', '.'))

    def on_tolerance(self, val):
        self.tolerance = float(val.replace(',', '.'))

    def analyse(self):
        im, delta = self.image_widget.get_image()
        mlc = self.mlc_model[self.comboBox.currentIndex()]
        QtGui.QMessageBox.information(None, "Information", mlc.name)
        im_iso = self.image_widget.isocenter
        mlc.isocenter = im_iso
        # settings
        conf = PicketFenceSettings(im, im_iso, delta, separation=self.separation, strip_size=self.strip_size,
                                   nstrips=self.n_strips,
                                   tolerance=self.tolerance)

        pf_test = PicketFenceTest(mlc, im, delta, im_iso, conf)
        pf_test.analyse()
        pf_test.prepare_result()
        self.fig_result, ax_result = pf_test.plot_leafs_results()
        _, ax_picket = pf_test.plot_picket_fence(self.image_widget.fig, self.image_widget.ax)

        self.picket_fence_test = pf_test
        self.calculated = True
        self.image_widget.update_image(self.image_widget.fig, ax_picket)
        self.result_widget.set_figure(self.fig_result)
        self.result_widget.show_figure()
        self.result_widget.showMaximized()

    def on_save(self):
        if self.calculated:
            file_name, _ = QtGui.QFileDialog.getSaveFileName(None, "Save Film2Dose Picket Fence file",
                                                             QtCore.QDir.currentPath(),
                                                             "Film2Dose Picket Fence files (*.pkt)")

            # save(self.result_data, file_name)
            save(self.picket_fence_test, file_name)
        else:
            QtGui.QMessageBox.information(None, "Information", "You need to analyse first")


class PicketFenceResultWidget(QtGui.QWidget):
    def __init__(self, parent=None):
        super(PicketFenceResultWidget, self).__init__(parent)
        self.fig = None
        self.canvas = None
        self.navigation_toolbar = None
        self.setWindowTitle(
            QtGui.QApplication.translate("Film2Dose", "Picket Fence Results", None, QtGui.QApplication.UnicodeUTF8))
        self.verticalLayout = QtGui.QVBoxLayout(self)
        self.setLayout(self.verticalLayout)

    def set_figure(self, fig):
        self.fig = fig

    def show_figure(self):
        self.canvas = FigureCanvas(self.fig)
        FigureCanvas.setSizePolicy(self, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.verticalLayout.addWidget(self.canvas)
        self.navigation_toolbar = Film2DoseToolbar(self.canvas, self)
        self.navigation_toolbar.setIconSize(QtCore.QSize(46, 46))
        self.verticalLayout.addWidget(self.navigation_toolbar)


class GammaComparisonWidget(QtGui.QWidget, DoseCompQT.Ui_DoseComp):
    def __init__(self, parent=None):
        super(GammaComparisonWidget, self).__init__(parent)
        self.setupUi(self)
        self.setWindowTitle(
            QtGui.QApplication.translate("Form", "Gamma Index Analysis", None, QtGui.QApplication.UnicodeUTF8))

        self.computed = np.array([])
        self.film = np.array([])
        self.registered = None
        self.G = np.array([])
        self.flat_G = np.array([])
        self.fig = None
        self.canvas = None
        self.ax = None
        self.reg_delta = None
        self.tps_widget = TPSWidget()
        self.film_widget = OptimizedDoseWidget()
        self.fusion_object = None
        self.gamma_widget = None
        self.scaled = False
        self.is_manual_scaled = False
        self.manual_msg = ''
        self.is_fused = False
        self.fus_msg = ''
        self.set_conections()

    def _on_import(self):
        self.tps_widget.read_image()
        self.film_widget.read_image()
        self.set_images()
        self._show_widgets()

    def set_images(self):
        self.gridLayout.addWidget(self.tps_widget, 0, 0, 1, 1)
        self.gridLayout.addWidget(self.film_widget, 0, 1, 1, 1)

    def set_conections(self):
        self.compare_button.clicked.connect(self.calculate_gamma)
        self.open_images.clicked.connect(self.on_open)
        self.save_images.clicked.connect(self.on_save)

    def on_save(self):
        if self.registered is not None:
            file_name, _ = QtGui.QFileDialog.getSaveFileName(None, "Film2Dose comparisson files",
                                                             QtCore.QDir.currentPath(),
                                                             "Film2Dose comparisson (*.cmp)")

            save(self.registered, file_name)

        else:
            QtGui.QMessageBox.information(None, "Information", "You have to register images first")

    def on_open(self):
        reply = QtGui.QMessageBox.question(self, "Image Registration",
                                           "Do you want to register both TPS and Film dose matrices",
                                           QtGui.QMessageBox.Yes | QtGui.QMessageBox.No | QtGui.QMessageBox.Cancel)

        if reply == QtGui.QMessageBox.Yes:
            self._on_import()
        elif reply == QtGui.QMessageBox.No:
            self.read_registered()

    def read_registered(self):

        file_location, pn = QtGui.QFileDialog.getOpenFileName(self,
                                                              "Import Film2Dose comparison files. *.cmp",
                                                              QtCore.QDir.currentPath(),
                                                              "Film2Dose Registered images (*.cmp);;")

        QtCore.QDir.setCurrent(file_location)

        if file_location:
            data = load(file_location)
            self.computed, self.film, self.reg_delta = data.get_registered()
            self.is_fused = True
            self._update_widgets()
            self.set_images()
            self._show_widgets()

    def _update_gamma(self):
        try:
            self.gridLayout.removeWidget(self.gamma_widget, 1, 0, 1, 1)
            self.gamma_widget.setParent(None)
            self.gridLayout.removeWidget(self.canvas, 1, 1, 1, 1)
            self.canvas.setParent(None)
        except:
            pass

    def _set_calculations(self):
        # get gamma index parameters
        dd = self.dosePercentageDoubleSpinBox.value() / 100.0
        # FIX DTA transform from mm to pixel space
        dta = round(float(self.dTAMmSpinBox.value()) / self.reg_delta)
        dt = self.doseThresholdDoubleSpinBox.value() / 100.0
        local = not self.local_checkBox.isChecked()
        # calculate_integrate the gamma index
        g = gamma_index(self.computed, self.film, dta, dd, dt, local)
        self.G = np.dstack((g, g, g))
        self.flat_G = np.ravel(g)
        self._gamma_results()

    def _update_widgets(self):
        self.tps_widget.set_image(self.computed, self.reg_delta)
        self.film_widget.set_image(np.dstack((self.film, self.film, self.film)), self.reg_delta)

    def _show_widgets(self):
        self.tps_widget.show_image()
        self.film_widget.show_image()

    def _reg_and_calc(self):
        f_dose, delta = self.film_widget.get_image()
        tps_dose, tps_delta = self.tps_widget.get_image()
        self.fusion_object = ImageRegistration(tps_dose, tps_delta, f_dose, delta)
        # self.computed, self.film, self.reg_delta = self.fusion_object.scaled_images()

        # show registration via normalized cross correlation
        if not self.is_fused:
            self.computed, self.film, self.reg_delta, flag, self.fus_msg = self.fusion_object.auto_registration()
            self.registered = FusedImages(self.computed, self.film, self.reg_delta)

            if flag:
                self._set_calculations()
            else:
                QtGui.QMessageBox.critical(self, "Error", self.fus_msg, QtGui.QMessageBox.Escape)

        else:
            self._set_calculations()

    def _gamma_results(self):
        channel = self.film_widget.channel
        self.gamma_widget.set_image(self.G, self.reg_delta, channel, 'Gamma')
        self.gamma_widget.show_image()
        self.gridLayout.addWidget(self.gamma_widget, 1, 0, 1, 1)

        # calculate_integrate the histogram.
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        idx = np.nonzero(self.flat_G)
        nzero = self.flat_G[idx]
        mask = nzero < 1  # Gamma index criterion
        pp = sum(mask) / float(mask.shape[0])
        val = round(pp, 3) * 100
        txt = 'Gamma index Histogram - points approved: %.2f' % val
        self.ax.set_title(txt)
        # the histogram of the data
        self.ax.hist(nzero, 50, normed=1, facecolor='green', alpha=0.75)
        self.gamma_hist = GammaHistogramWidget(self.fig)
        self.gridLayout.addWidget(self.gamma_hist, 1, 1, 1, 1)

    def calculate_gamma(self):
        self._update_gamma()
        self.gamma_widget = EditImageWidget()
        self._reg_and_calc()


class DoseComparisonWidget(QtGui.QWidget, DoseCompAbsQT.Ui_DoseCompAbs):
    def __init__(self, parent=None):
        super(DoseComparisonWidget, self).__init__(parent)
        self.setupUi(self)
        self.setWindowTitle(
            QtGui.QApplication.translate("Form", "Gamma Index Analysis", None, QtGui.QApplication.UnicodeUTF8))

        self.computed = np.array([])
        self.film = np.array([])
        self.registered = None
        self.G = np.array([])
        self.flat_G = np.array([])
        self.fig = None
        self.canvas = None
        self.ax = None
        self.reg_delta = None
        self.tps_widget = TPSWidget()
        self.film_widget = OptimizedDoseWidget()
        self.fusion_object = None
        self.result_widget = None
        self.mode = 1  # 1 : auto , 0 : manual
        self.scaled = False
        self.is_manual_scaled = False
        self.manual_msg = ''
        self.is_fused = False
        self.fus_msg = ''
        self.set_conections()

    def set_conections(self):
        self.open_images.clicked.connect(self.on_open)
        self.compare_button.clicked.connect(self.calc_diff)
        self.abs_radio.clicked.connect(self.on_abs)
        self.rel_radio.clicked.connect(self.on_rel)
        self.profiles_button.clicked.connect(self.on_profiles)
        self.save_images.clicked.connect(self.on_save)

    def _on_import(self):
        self.tps_widget.read_image()
        self.film_widget.read_image()
        self.set_images()
        self._show_widgets()

    def on_open(self):
        reply = QtGui.QMessageBox.question(self, "Image Registration",
                                           "Do you want to register both TPS and Film dose matrices",
                                           QtGui.QMessageBox.Yes | QtGui.QMessageBox.No | QtGui.QMessageBox.Cancel)

        if reply == QtGui.QMessageBox.Yes:
            self._on_import()
        elif reply == QtGui.QMessageBox.No:
            self.read_registered()

    def read_registered(self):

        file_location, pn = QtGui.QFileDialog.getOpenFileName(self,
                                                              "Import Film2Dose comparison files. *.cmp",
                                                              QtCore.QDir.currentPath(),
                                                              "Film2Dose Registered images (*.cmp);;")

        QtCore.QDir.setCurrent(file_location)

        if file_location:
            data = load(file_location)
            self.computed, self.film, self.reg_delta = data.get_registered()
            self.is_fused = True
            self._update_widgets()
            self.set_images()
            self._show_widgets()

    def on_save(self):
        if self.registered is not None:
            file_name, _ = QtGui.QFileDialog.getSaveFileName(None, "Film2Dose comparisson files",
                                                             QtCore.QDir.currentPath(),
                                                             "Film2Dose comparisson (*.cmp)")

            save(self.registered, file_name)

        else:
            QtGui.QMessageBox.information(None, "Information", "You have to register images first")

    def set_images(self):
        self.gridLayout.addWidget(self.tps_widget, 0, 0, 1, 1)
        self.gridLayout.addWidget(self.film_widget, 0, 1, 1, 1)

    def on_abs(self):
        self.mode = 0

    def on_rel(self):
        self.mode = 1

    def _update_gamma(self):
        try:
            self.gridLayout.removeWidget(self.result_widget, 1, 0, 1, 1)
            self.result_widget.setParent(None)
            self.gridLayout.removeWidget(self.canvas, 1, 1, 1, 1)
            self.canvas.setParent(None)
        except:
            pass

    def _update_widgets(self):
        self.tps_widget.set_image(self.computed, self.reg_delta)
        self.film_widget.set_image(np.dstack((self.film, self.film, self.film)), self.reg_delta)

    def _show_widgets(self):
        self.tps_widget.show_image()
        self.film_widget.show_image()

    def _reg(self):
        if not self.is_fused:
            f_dose, delta = self.film_widget.get_image()
            tps_dose, tps_delta = self.tps_widget.get_image()
            self.fusion_object = ImageRegistration(tps_dose, tps_delta, f_dose, delta)
            self.computed, self.film, self.reg_delta, flag, self.fus_msg = self.fusion_object.auto_registration()
            self.registered = FusedImages(self.computed, self.film, self.reg_delta)

    def on_profiles(self):
        self._update_gamma()
        self._reg()
        self.result_widget = ProfileComparison(self.computed, self.film, self.reg_delta)
        plt.show()

    def _reg_and_calc(self):
        self._reg()
        if self.mode == 1:
            # Relative difference
            g = dose_difference(self.computed, self.film, pp=self.mode)
            self.G = np.dstack((g, g, g))
            self.flat_G = np.ravel(g)
            self._calc_results()
        elif self.mode == 0:
            # absolute difference
            g = dose_difference(self.computed, self.film, pp=self.mode)
            self.G = np.dstack((g, g, g))
            self.flat_G = np.ravel(g)
            self._calc_results()

    def _calc_results(self):
        channel = self.film_widget.channel
        self.result_widget.set_image(self.G, self.reg_delta, channel, 'Gamma Index')
        self.result_widget.show_image()
        self.gridLayout.addWidget(self.result_widget, 1, 0, 1, 1)

        # calculate_integrate the histogram.
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        m = self.flat_G.mean()
        st = self.flat_G.std(ddof=1)
        txt = 'Dose Difference Histogram - Mean: %.2f STD: %.2f' % (m, st)
        self.ax.set_title(txt)
        # the histogram of the data

        self.ax.hist(self.flat_G, 50, normed=1, facecolor='green', alpha=0.75)
        self.canvas = FigureCanvas(self.fig)
        self.gridLayout.addWidget(self.canvas, 1, 1, 1, 1)

    def calc_diff(self):
        self._update_gamma()
        self.result_widget = EditImageWidget()
        self._reg_and_calc()


class EditImageWidget(QtGui.QWidget, FormImageQT.Ui_imageForm):
    def __init__(self, parent=None):
        super(EditImageWidget, self).__init__(parent)

        self.setupUi(self)
        self.image_location = None
        self.calib_data = {}
        self.channel = 0
        self.channel_names = ['Red channel', 'Green channel', 'Blue channel', 'Disturbance', 'Residues map']
        self.im = np.array([])
        self.im_bkp = np.array([])
        self.min_sat = None
        self.max_sat = None
        self.delta = 0.0
        self.canvas = None
        self.fig = None
        self.ax = None
        self.cursor = None
        self.navigation_toolbar = None
        self.isocenter = np.array([0, 0], dtype=float)
        self.cursor_position = np.array([0, 0])
        self.image_title = ""
        self.image_type_names = {-1: "Optical Density", 0: "Dose (cGy)", 1: "Pixel Value", 2: 'Gamma Matrix'}
        self.image_type = ''
        self.colormap = 'jet'
        self.cal = None
        self.showed = False
        self.scale = 'mm'
        # colormaps
        self.cmaps = ['jet', 'viridis', 'Blues', 'BuGn', 'BuPu', 'GnBu', 'Greens', 'Greys', 'Oranges', 'OrRd', 'PuBu',
                      'PuBuGn', 'PuRd', 'Purples', 'RdPu', 'Reds', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd', 'afmhot',
                      'autumn', 'bone', 'cool', 'copper', 'gist_heat', 'gray', 'hot', 'pink', 'spring', 'summer',
                      'winter', 'BrBG', 'bwr', 'coolwarm', 'PiYG', 'PRGn', 'PuOr', 'RdBu', 'RdGy', 'RdYlBu', 'RdYlGn',
                      'Spectral', 'seismic', 'Accent', 'Dark2', 'Paired', 'Pastel1', 'Pastel2', 'Set1', 'Set2', 'Set3',
                      'gist_earth', 'terrain', 'ocean', 'gist_stern', 'brg', 'CMRmap', 'cubehelix', 'gnuplot',
                      'gnuplot2', 'gist_ncar', 'nipy_spectral', 'rainbow', 'gist_rainbow', 'hsv', 'flag', 'prism']
        self._colormap_combo()
        self.window_validator()
        self.set_connections()

    def window_validator(self):
        # integers 0 to 9999
        rx = QtCore.QRegExp("[0-9]\\d{0,3}")
        # the validator treats the regexp as "^[1-9]\\d{0,3}$"
        v = QtGui.QRegExpValidator()
        v.setRegExp(rx)
        self.minLineEdit.setValidator(v)
        self.maxLineEdit.setValidator(v)

    @property
    def iso_reg(self):
        return self.cursor_position

    def set_scale(self, scale='mm'):
        self.scale = scale

    def _colormap_combo(self):
        for item in self.cmaps:
            self.colorComboBox.addItem(item)

    def set_connections(self):
        self.channel_box.activated.connect(self.on_activated)
        self.colorComboBox.activated[str].connect(self.on_color)
        self.minLineEdit.returnPressed.connect(self.on_min)
        self.maxLineEdit.returnPressed.connect(self.on_max)
        self.rotate_90cw.clicked.connect(self.on_rotateCW)
        self.rotate_90ccw.clicked.connect(self.on_rotateCCW)
        self.button_rotatePoints.clicked.connect(self.on_rotate)
        self.save_as.clicked.connect(self.save_images)
        self.isocenter_button.clicked.connect(self.set_isocenter)
        self.button_fliplr.clicked.connect(self.on_fliplr)
        self.button_flipud.clicked.connect(self.on_flipud)

    def set_image(self, im, delta, channel=1, im_type='', calib_data=None):
        self.im = im
        self.delta = delta
        self.channel = channel
        self.image_type = im_type
        self.calib_data = calib_data
        self.im_bkp = self.im.copy()

    def set_colormap(self, colormap='jet'):
        """
            Set the colormap of the EditImageWidget.
        colormap = [('Sequential',     ['Blues', 'BuGn', 'BuPu',
                             'GnBu', 'Greens', 'Greys', 'Oranges', 'OrRd',
                             'PuBu', 'PuBuGn', 'PuRd', 'Purples', 'RdPu',
                             'Reds', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd']),
         ('Sequential (2)', ['afmhot', 'autumn', 'bone', 'cool', 'copper',
                             'gist_heat', 'gray', 'hot', 'pink',
                             'spring', 'summer', 'winter']),
         ('Diverging',      ['BrBG', 'bwr', 'coolwarm', 'PiYG', 'PRGn', 'PuOr',
                             'RdBu', 'RdGy', 'RdYlBu', 'RdYlGn', 'Spectral',
                             'seismic']),
         ('Qualitative',    ['Accent', 'Dark2', 'Paired', 'Pastel1',
                             'Pastel2', 'Set1', 'Set2', 'Set3']),
         ('Miscellaneous',  ['gist_earth', 'terrain', 'ocean', 'gist_stern',
                             'brg', 'CMRmap', 'cubehelix',
                             'gnuplot', 'gnuplot2', 'gist_ncar',
                             'nipy_spectral',  'viridis', 'jet', 'rainbow',
                             'gist_rainbow', 'hsv', 'flag', 'prism'])
        :param colormap: str of colormap
        """
        self.colormap = colormap

    def set_image_path(self, path):
        self.image_location = path

    def read_image(self, path_to_image=None):

        if path_to_image is None:
            self.image_location, pn = QtGui.QFileDialog.getOpenFileName(self,
                                                                        "Import 48 bits tiff File or Film2Dose image files.",
                                                                        QtCore.QDir.currentPath(),
                                                                        "Tiff Files (*.tif);;"
                                                                        "Film2Dose images (*.fti);;"
                                                                        "Film2Dose Dose images (*.ftd);;"
                                                                        "DICOM Images (*.dcm)")

        else:
            self.image_location = path_to_image

        QtCore.QDir.setCurrent(self.image_location)

        _, filepart = os.path.splitext(self.image_location)

        if self.image_location:
            if filepart == '.tif':
                data, self.delta = read_tiff(self.image_location)
                self.image_type = 'tif'
                self.image_title = self.image_type_names[-1]
                self.im = np.zeros(data.shape)
                self.im[:, :, 0] = data[:, :, 0]
                self.im[:, :, 1] = data[:, :, 1]
                self.im[:, :, 2] = data[:, :, 2]

            elif filepart == '.fti':
                data = load(self.image_location)
                self.im, self.delta, self.image_type, self.calib_data = data.get_image()
                if self.image_type == "Pixel":
                    self.image_title = self.image_type_names[1]
                else:
                    self.image_title = self.image_type_names[-1]

            elif filepart == '.ftd':
                data = load(self.image_location)
                self.im, self.delta, self.image_type, self.calib_data = data.get_image()
                self.image_title = self.image_type_names[0]

            elif filepart == '.dcm':
                im, self.delta = read_dicom(self.image_location)
                self.im = np.zeros((im.shape[0], im.shape[1], 3))
                self.im[:, :, 0] = im
                self.im[:, :, 1] = im
                self.im[:, :, 2] = im
                self.image_type = 'DICOM'

        self.im_bkp = self.im.copy()

    def set_windows_limits(self, im):
        try:
            self.min_sat = np.percentile(im, 1)
            self.max_sat = np.percentile(im, 99)
        except:
            self.min_sat = im.min()
            self.max_sat = im.max()
        mi = str(round(self.min_sat))
        mx = str(round(self.max_sat))
        self.minLineEdit.setText(mi)
        self.maxLineEdit.setText(mx)

    def show_image(self, fig=None, ax=None):
        if fig is None and ax is None:
            im = self.im[:, :, self.channel]
            if self.min_sat is None or self.max_sat is None:
                self.set_windows_limits(im)
                if self.image_type == 'Gamma':
                    self.min_sat = im.min()
                    self.max_sat = 1.0
                    self.minLineEdit.setText(str(0))
                    self.maxLineEdit.setText(str(1))

                self.fig, self.ax = display_fig(im=im, delta=self.delta, col_map=self.colormap,
                                                limits=(self.min_sat, self.max_sat),
                                                offset=self.isocenter, scale=self.scale)
        elif fig == 1:
            im = self.im[:, :, self.channel]
            self.set_windows_limits(im)
            lim = (self.min_sat, self.max_sat)
            del self.fig
            del self.ax
            self.fig, self.ax = display_fig(im, self.delta, self.colormap, lim, self.isocenter, self.scale)

        else:
            self.fig = fig
            self.ax = ax

        if self.image_type == 'tif':
            title = self.image_type_names[-1] + " - " + self.channel_names[self.channel]
        elif self.image_type == 'Gamma':
            title = 'Gamma  Matrix'
        else:
            title = self.image_type_names[self.channel] + " - " + self.channel_names[self.channel]

        self.ax.set_title(title)
        self.canvas = RotationCanvas(self.fig)
        self.verticalLayout_2.addWidget(self.canvas)
        self.navigation_toolbar = Film2DoseToolbar(self.canvas, self)
        self.navigation_toolbar.setIconSize(QtCore.QSize(46, 46))
        self.verticalLayout_2.addWidget(self.navigation_toolbar)

    def image_threshold(self):

        im = auto_threshold(self.im[:, :, self.channel])
        self.im[:, :, self.channel] = im
        self.update_image(fig=1)

    def image_trim(self, x_border, y_border):
        self.im = image_trim_xy(self.im, self.delta, x_border, y_border)
        self.update_image(fig=1)

    def restore_image(self):
        self.im = self.im_bkp.copy()
        self.update_image(fig=1)

    def on_color(self, txt):
        self.colormap = txt
        self.update_image(fig=1)

    def on_min(self):
        print('on_min')
        self.min_sat = float(self.minLineEdit.text())
        self.update_image(fig=1)

    def on_max(self):
        print('on_max')
        self.max_sat = float(self.maxLineEdit.text())
        self.update_image(fig=1)

    def get_canvas_points(self, n):
        self.update_image()
        self.cursor = Cursor(self.ax, useblit=True, color='red', linewidth=1)
        self.canvas.set_points(n)
        tmp = self.canvas.get_points()
        self.cursor.disconnect_events()
        pos = np.asarray(tmp[0])
        return pos

    def on_flipud(self):
        # TODO add a header to keep all data manipulations.
        self.im = np.flipud(self.im)
        self.update_image(fig=1)

    def on_fliplr(self):
        self.im = np.fliplr(self.im)
        self.update_image(fig=1)

    def get_position(self):
        self.update_image()
        self.cursor = Cursor(self.ax, useblit=True, color='red', linewidth=1)
        self.canvas.set_points(1)
        position = self.canvas.get_points()
        self.update_image(fig=1)
        return np.asarray(position[0])

    def get_points(self, npoints):
        self.update_image()
        self.cursor = Cursor(self.ax, useblit=True, color='red', linewidth=1)
        self.canvas.set_points(npoints)
        position = self.canvas.get_points()
        self.update_image(fig=1)
        return np.asarray(position)

    def set_isocenter(self):
        self.update_image()
        self.cursor = Cursor(self.ax, useblit=True, color='red', linewidth=1)
        self.canvas.set_points(1)
        position = self.canvas.get_points()
        pos = np.array(position[0], dtype=float)
        self.cursor_position = np.asarray(position[0]).astype(int)
        self.isocenter += pos
        print(self.cursor_position)
        print('Position: ', self.cursor_position)
        print('actual isocenter: ', self.isocenter)
        self.update_image(fig=1)

    def save_images(self):

        h0, h1 = self.ax.get_xlim()
        v0, v1 = self.ax.get_ylim()
        imc = get_crop(self.im, self.delta, [h0, h1, v0, v1])
        print('limits: xlim: %s, %s  ylim: %s, %s' % (h0, h1, v0, v1))
        # print(self.delta)
        im = Fim2DoseImage(imc, self.delta, self.image_type, self.isocenter, self.calib_data)
        file_name, _ = QtGui.QFileDialog.getSaveFileName(None, "Save Film2Dose image",
                                                         QtCore.QDir.currentPath(),
                                                         "Film2Dose images (*.fti)")

        if file_name[-3:] == 'fti':
            save_ftd(im, file_name)

    def get_image(self):
        return self.im[:, :, self.channel], self.delta

    def get_all_channels(self):
        return self.im, self.delta

    def on_rotateCW(self):
        self.im = np.rot90(self.im, 3)
        self.update_image(fig=1)

    def on_rotateCCW(self):
        self.im = np.rot90(self.im)
        self.update_image(fig=1)

    def on_rotate(self):
        self.cursor = Cursor(self.ax, useblit=True, color='red', linewidth=1)
        position = self.canvas.get_points()
        x = (position[0][0], position[1][0])
        y = (position[0][1], position[1][1])
        self.im = rotate_image(self.im, x, y)
        self.update_image(fig=1)

    def update_image(self, fig=None, ax=None):
        try:
            self.verticalLayout_2.removeWidget(self.canvas)
            self.canvas.setParent(None)
            self.verticalLayout_2.removeWidget(self.navigation_toolbar)
            self.navigation_toolbar.setParent(None)
            del self.canvas
            del self.navigation_toolbar
            self.show_image(fig, ax)
        except:
            pass

    def on_activated(self):

        if self.channel_box.currentIndex() == 0:
            self.update_combo()
        elif self.channel_box.currentIndex() == 1:
            self.update_combo()
        elif self.channel_box.currentIndex() == 2:
            self.update_combo()

    def update_combo(self):
        self.channel = self.channel_box.currentIndex()
        try:
            self.verticalLayout_2.removeWidget(self.canvas)
            self.canvas.setParent(None)
            self.verticalLayout_2.removeWidget(self.navigation_toolbar)
            self.navigation_toolbar.setParent(None)
            self.show_image()
        except:
            pass


class TPSWidget(QtGui.QWidget, TPSWidgetQT.Ui_imageForm):
    # TODO refactor dose window using line edit
    def __init__(self, parent=None):
        super(TPSWidget, self).__init__(parent)
        self.setupUi(self)
        self.image_location = None
        self.im = np.array([])
        self.image_type = ''
        self.min_sat = None
        self.max_sat = None
        self.delta = 0.0
        self.canvas = None
        self.fig = None
        self.ax = None
        self.cursor = None
        self.navigation_toolbar = None
        self.isocenter = np.array([0.0, 0.0])
        self.cursor_position = np.array([0.0, 0.0])
        self.image_title = 'TPS calculated doses - cGy'
        self.colormap = 'jet'
        self.radio_mm.setChecked(True)
        self.scale = 'mm'
        self.interpolation = None
        # colormaps
        self.cmaps = ['jet', 'viridis', 'Blues', 'BuGn', 'BuPu', 'GnBu', 'Greens', 'Greys', 'Oranges', 'OrRd', 'PuBu',
                      'PuBuGn',
                      'PuRd', 'Purples', 'RdPu', 'Reds', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd', 'afmhot', 'autumn',
                      'bone', 'cool', 'copper', 'gist_heat', 'gray', 'hot', 'pink', 'spring', 'summer', 'winter',
                      'BrBG', 'bwr', 'coolwarm', 'PiYG', 'PRGn', 'PuOr', 'RdBu', 'RdGy', 'RdYlBu', 'RdYlGn', 'Spectral',
                      'seismic', 'Accent', 'Dark2', 'Paired', 'Pastel1', 'Pastel2', 'Set1', 'Set2', 'Set3',
                      'gist_earth', 'terrain', 'ocean', 'gist_stern', 'brg', 'CMRmap', 'cubehelix', 'gnuplot',
                      'gnuplot2', 'gist_ncar', 'nipy_spectral', 'rainbow', 'gist_rainbow', 'hsv', 'flag', 'prism']
        self._colormap_combo()
        self.window_validator()
        self.set_connections()

    def window_validator(self):
        # integers 0 to 9999
        rx = QtCore.QRegExp("[0-9]\\d{0,3}")
        # the validator treats the regexp as "^[1-9]\\d{0,3}$"
        v = QtGui.QRegExpValidator()
        v.setRegExp(rx)
        self.minLineEdit.setValidator(v)
        self.maxLineEdit.setValidator(v)

    def get_points(self, npoints):
        self.update_image()
        self.cursor = Cursor(self.ax, useblit=True, color='red', linewidth=1)
        self.canvas.set_points(npoints)
        position = self.canvas.get_points()
        self.update_image()
        return np.asarray(position)

    def set_image(self, im, delta):
        self.im = im
        self.delta = delta

    def get_image(self):
        return self.im, self.delta

    @property
    def iso_reg(self):
        return self.cursor_position

    def set_scale(self, scale='mm'):
        self.scale = scale

    def _colormap_combo(self):
        for item in self.cmaps:
            self.colorComboBox.addItem(item)

    def set_connections(self):
        self.colorComboBox.activated[str].connect(self.on_color)
        self.minLineEdit.returnPressed.connect(self.on_min)
        self.maxLineEdit.returnPressed.connect(self.on_max)
        self.rotate_90cw.clicked.connect(self.on_rotateCW)
        self.rotate_90ccw.clicked.connect(self.on_rotateCCW)
        self.open_button.clicked.connect(self.on_open)
        self.button_fliplr.clicked.connect(self.on_fliplr)
        self.button_flipud.clicked.connect(self.on_flipud)
        self.multiply_button.clicked.connect(self.on_multiply)
        self.radio_mm.clicked.connect(self.on_mm)
        self.radio_pixel.clicked.connect(self.on_pixel)

    def on_mm(self):
        self.scale = 'mm'
        self.update_image()

    def on_pixel(self):
        self.scale = 'pix'
        self.update_image()

    def on_multiply(self):
        """
            Multipy image by a factor ( float number)

        """
        factor, flag = QtGui.QInputDialog.getDouble(self, "Multiply image by a factor", "factor", decimals=6, value=1)
        if flag:
            self.im = self.im * factor
            self.update_image()

    def on_open(self):
        flag = self.read_image()
        if flag:
            self.update_image()

    def on_color(self, txt):
        self.colormap = txt
        self.update_image()

    def on_min(self):
        print('on_min')
        self.min_sat = float(self.minLineEdit.text())
        self.update_image(fig=1)

    def on_max(self):
        print('on_max')
        self.max_sat = float(self.maxLineEdit.text())
        self.update_image(fig=1)

    def on_rotateCW(self):
        self.im = np.rot90(self.im, 3)
        self.update_image()

    def on_rotateCCW(self):
        self.im = np.rot90(self.im)
        self.update_image()

    def get_canvas_points(self, n):
        self.update_image()
        self.cursor = Cursor(self.ax, useblit=True, color='red', linewidth=1)
        self.canvas.set_points(n)
        tmp = self.canvas.get_points()
        self.cursor.disconnect_events()
        pos = np.asarray(tmp[0])
        return pos

    def on_flipud(self):
        self.im = np.flipud(self.im)
        self.update_image()

    def on_fliplr(self):
        self.im = np.fliplr(self.im)
        self.update_image()

    def set_isocenter(self):
        self.update_image()
        self.cursor = Cursor(self.ax, useblit=True, color='red', linewidth=1)
        self.canvas.set_points(1)
        position = self.canvas.get_points()
        self.cursor_position = np.asarray(position[0]).astype(int)
        self.isocenter += np.asarray(position[0])
        print('Position: ' + str(self.cursor_position))
        print('actual isocenter: ' + str(self.isocenter))
        self.update_image()

    def set_colormap(self, colormap='jet'):
        """
            Set the colormap of the EditImageWidget.
        colormap = [('Sequential',     ['Blues', 'BuGn', 'BuPu',
                             'GnBu', 'Greens', 'Greys', 'Oranges', 'OrRd',
                             'PuBu', 'PuBuGn', 'PuRd', 'Purples', 'RdPu',
                             'Reds', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd']),
         ('Sequential (2)', ['afmhot', 'autumn', 'bone', 'cool', 'copper',
                             'gist_heat', 'gray', 'hot', 'pink',
                             'spring', 'summer', 'winter']),
         ('Diverging',      ['BrBG', 'bwr', 'coolwarm', 'PiYG', 'PRGn', 'PuOr',
                             'RdBu', 'RdGy', 'RdYlBu', 'RdYlGn', 'Spectral',
                             'seismic']),
         ('Qualitative',    ['Accent', 'Dark2', 'Paired', 'Pastel1',
                             'Pastel2', 'Set1', 'Set2', 'Set3']),
         ('Miscellaneous',  ['gist_earth', 'terrain', 'ocean', 'gist_stern',
                             'brg', 'CMRmap', 'cubehelix',
                             'gnuplot', 'gnuplot2', 'gist_ncar',
                             'nipy_spectral',  'viridis', 'jet', 'rainbow',
                             'gist_rainbow', 'hsv', 'flag', 'prism'])
        :param colormap: str of colormap
        """
        self.colormap = colormap

    def set_image_path(self, path):
        self.image_location = path

    def read_image(self, path_to_image=None):

        if path_to_image is None:
            self.image_location, pn = QtGui.QFileDialog.getOpenFileNames(self,
                                                                         "Import TPS Doses",
                                                                         QtCore.QDir.currentPath(),
                                                                         "DICOM RT (*.dcm);;"
                                                                         "Brainlab files (*.*);;"
                                                                         "MONACO/XiO Files (*.txt);;"
                                                                         "MONACO/XiO Files (*.1);;"
                                                                         "MONACO/XiO Files (*.All);;"
                                                                         "OmniPro I'mRT (IBA)(*.opg)")
        else:
            self.image_location = path_to_image
        # print(type(self.image_location), len(self.image_location))
        if self.image_location:
            QtCore.QDir.setCurrent(self.image_location[0])
            _, filepart = os.path.splitext(self.image_location[0])
            if self.image_location:
                if filepart == '.dcm':
                    tmp, self.delta = read_dicom(self.image_location[0])
                    self.im = np.zeros(tmp.shape)
                    for f in self.image_location:
                        tmp, self.delta = read_dicom(f)
                        self.im += tmp
                    self.image_type = 'DICOMRT'

                if filepart in ['.flu', '.dat', '.35MM1']:
                    tmp, self.delta = read_brainlab(self.image_location[0])
                    self.im = np.zeros(tmp.shape)
                    for f in self.image_location:
                        tmp, self.delta = read_brainlab(f)
                        self.im += tmp

                    self.image_type = 'Brainlab'

                if filepart in ['.1', '.All', '.txt', '.ALL']:
                    tmp, self.delta = read_monaco(self.image_location[0])
                    self.im = np.zeros(tmp.shape)
                    for f in self.image_location:
                        tmp, self.delta = read_monaco(f)
                        self.im += tmp
                    self.image_type = 'Monaco/XiO'

                if filepart in ['.opg']:
                    tmp, self.delta = read_OmniPro(self.image_location[0])
                    self.im = np.zeros(tmp.shape)
                    for f in self.image_location:
                        tmp, self.delta = read_OmniPro(f)
                        self.im += tmp

                    self.interpolation = 'bicubic'
                    self.image_type = "OmniPro I'mRT (IBA)"

                if filepart in ['.mcc']:
                    of = OctaviusFiles(self.image_location)
                    self.im, self.delta = of.get_data()
                    self.interpolation = 'bicubic'
                    self.image_type = "PTW Octavius"

                return True

        else:
            return False

    def show_image(self, fig=None, ax=None):

        if fig is None and ax is None:
            im = self.im
            self.min_sat = np.percentile(im, 1)
            self.max_sat = np.percentile(im, 99)
            mi = str(round(self.min_sat))
            mx = str(round(self.max_sat))
            self.minLineEdit.setText(mi)
            self.maxLineEdit.setText(mx)
            lim = (self.min_sat, self.max_sat)
            self.fig, self.ax = display_fig(im, self.delta, self.colormap, lim, self.isocenter, self.scale,
                                            interp=self.interpolation)
        elif fig == 1:
            im = self.im
            lim = (self.min_sat, self.max_sat)
            print('limit:', lim)
            self.fig, self.ax = display_fig(im, self.delta, self.colormap, lim, self.isocenter, self.scale,
                                            interp=self.interpolation)
        else:
            self.fig = fig
            self.ax = ax

        self.ax.set_title(self.image_title)
        self.canvas = RotationCanvas(self.fig)
        self.verticalLayout_2.addWidget(self.canvas)
        self.navigation_toolbar = Film2DoseToolbar(self.canvas, self)
        self.navigation_toolbar.setIconSize(QtCore.QSize(46, 46))
        self.verticalLayout_2.addWidget(self.navigation_toolbar)

    def update_image(self, fig=None, ax=None):
        try:
            self.verticalLayout_2.removeWidget(self.canvas)
            self.canvas.setParent(None)
            self.verticalLayout_2.removeWidget(self.navigation_toolbar)
            self.navigation_toolbar.setParent(None)
            del self.canvas
            del self.navigation_toolbar
            self.show_image(fig, ax)
        except:
            pass


class OptimizedDoseWidget(QtGui.QWidget, DoseOptimizedQT.Ui_DoseOptimForm):
    def __init__(self, parent=None):
        super(OptimizedDoseWidget, self).__init__(parent)
        self.setupUi(self)
        self.image_location = None
        self.calib_data = {}
        self.channel = 0
        # self.channel_names = ['Red channel', 'Green channel', 'Blue channel', 'Disturbance', 'Residues map']
        self.channel_names = ['Dose', 'Disturbance', 'Residues map']
        self.im = np.array([])
        self.min_sat = None
        self.max_sat = None
        self.delta = 0.0
        self.canvas = None
        self.fig = None
        self.ax = None
        self.cursor = None
        self.navigation_toolbar = None
        self.isocenter = np.array([0.0, 0.0])
        self.cursor_position = np.array([0.0, 0.0])
        self.image_title = ""
        self.image_type_names = {-1: "Optical Density", 0: "Dose (cGy)", 1: "Dose (cGy)", 2: "Dose (cGy)", 3: "%",
                                 4: "Dose (cGy)"}
        self.image_type = ''
        self.colormap = 'jet'
        self.cal = None
        self.showed = False
        self.scale = 'mm'
        self.calc_method = ''
        # colormaps
        self.cmaps = ['jet', 'viridis', 'Blues', 'BuGn', 'BuPu', 'GnBu', 'Greens', 'Greys', 'Oranges', 'OrRd', 'PuBu',
                      'PuBuGn',
                      'PuRd', 'Purples', 'RdPu', 'Reds', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd', 'afmhot', 'autumn',
                      'bone', 'cool', 'copper', 'gist_heat', 'gray', 'hot', 'pink', 'spring', 'summer', 'winter',
                      'BrBG', 'bwr', 'coolwarm', 'PiYG', 'PRGn', 'PuOr', 'RdBu', 'RdGy', 'RdYlBu', 'RdYlGn', 'Spectral',
                      'seismic', 'Accent', 'Dark2', 'Paired', 'Pastel1', 'Pastel2', 'Set1', 'Set2', 'Set3',
                      'gist_earth', 'terrain', 'ocean', 'gist_stern', 'brg', 'CMRmap', 'cubehelix', 'gnuplot',
                      'gnuplot2', 'gist_ncar', 'nipy_spectral', 'rainbow', 'gist_rainbow', 'hsv', 'flag', 'prism']
        self._colormap_combo()
        self.window_validator()
        self.set_connections()

    def window_validator(self):
        # integers 0 to 9999
        rx = QtCore.QRegExp("[0-9]\\d{0,3}")
        # the validator treats the regexp as "^[1-9]\\d{0,3}$"
        v = QtGui.QRegExpValidator()
        v.setRegExp(rx)
        self.minLineEdit.setValidator(v)
        self.maxLineEdit.setValidator(v)

    def set_method(self, method):
        self.calc_method = method

    @property
    def iso_reg(self):
        return self.cursor_position

    def set_scale(self, scale='mm'):
        self.scale = scale

    def _colormap_combo(self):
        for item in self.cmaps:
            self.colorComboBox.addItem(item)

    def set_connections(self):
        self.channel_box.activated.connect(self.on_activated)
        self.colorComboBox.activated[str].connect(self.on_color)
        self.minLineEdit.returnPressed.connect(self.on_min)
        self.maxLineEdit.returnPressed.connect(self.on_max)
        self.rotate_90cw.clicked.connect(self.on_rotateCW)
        self.rotate_90ccw.clicked.connect(self.on_rotateCCW)
        self.button_rotatePoints.clicked.connect(self.on_rotate)
        self.open_button.clicked.connect(self.on_open)
        self.save_as.clicked.connect(self.save_images)
        self.isocenter_button.clicked.connect(self.set_isocenter)
        self.button_fliplr.clicked.connect(self.on_fliplr)
        self.button_flipud.clicked.connect(self.on_flipud)
        # Dosimetry button
        self.button_pointDose.clicked.connect(self.on_point_dose)
        self.multiply_button.clicked.connect(self.on_multiply)
        self.radio_mm.clicked.connect(self.on_mm)
        self.radio_pixel.clicked.connect(self.on_pixel)

    def on_mm(self):
        self.scale = 'mm'
        self.update_image()

    def on_pixel(self):
        self.scale = 'pix'
        self.update_image()

    def on_open(self):
        flag = self.read_image()
        if flag:
            self.update_image()

    def on_point_dose(self):

        if self.image_type == 'ftd-dose':
            self.update_image()
            self.cursor = Cursor(self.ax, useblit=True, color='red', linewidth=1)
            self.canvas.set_points(1)
            tmp = self.canvas.get_points()
            self.cursor.disconnect_events()
            pos = np.asarray(tmp[0])
            pos[1] *= -1  # Correct Y coordinate
            # TODO why do I need to flip the image up-down ? Debug image_crop ?
            dose_im = image_crop(self.im, self.delta, pos)
            V, dose = get_covarmatrix(self.calib_data, dose_im, self.delta, self.channel)
            type_B, sigma, Npix = analyse_roi(dose_im[:, :, 2])
            txt = meanval_uncertainty(dose, V, type_B)
            QtGui.QMessageBox.information(None, "Information", txt)
        else:
            if not self.showed:
                message = "<p>It is not possible to calculate_integrate uncertainty for type *.%s images</p>" % self.image_type
                QtGui.QMessageBox.information(None, "Information", message)
                self.showed = True

    def on_multiply(self):
        """
            Multipy image by a factor ( float number)

        """
        factor, flag = QtGui.QInputDialog.getDouble(self, "Multiply image by a factor", "factor", decimals=6, value=1)
        if flag:
            self.im[:, :, self.channel] = self.im[:, :, self.channel] * factor
            self.update_image()

    def set_image(self, im, delta, channel=1, im_type='', calib_data=None):
        self.im = im
        self.delta = delta
        self.channel = channel
        self.image_type = im_type
        self.calib_data = calib_data

    def set_colormap(self, colormap='jet'):
        """
            Set the colormap of the EditImageWidget.
        colormap = [('Sequential',     ['Blues', 'BuGn', 'BuPu',
                             'GnBu', 'Greens', 'Greys', 'Oranges', 'OrRd',
                             'PuBu', 'PuBuGn', 'PuRd', 'Purples', 'RdPu',
                             'Reds', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd']),
         ('Sequential (2)', ['afmhot', 'autumn', 'bone', 'cool', 'copper',
                             'gist_heat', 'gray', 'hot', 'pink',
                             'spring', 'summer', 'winter']),
         ('Diverging',      ['BrBG', 'bwr', 'coolwarm', 'PiYG', 'PRGn', 'PuOr',
                             'RdBu', 'RdGy', 'RdYlBu', 'RdYlGn', 'Spectral',
                             'seismic']),
         ('Qualitative',    ['Accent', 'Dark2', 'Paired', 'Pastel1',
                             'Pastel2', 'Set1', 'Set2', 'Set3']),
         ('Miscellaneous',  ['gist_earth', 'terrain', 'ocean', 'gist_stern',
                             'brg', 'CMRmap', 'cubehelix',
                             'gnuplot', 'gnuplot2', 'gist_ncar',
                             'nipy_spectral',  'viridis', 'jet', 'rainbow',
                             'gist_rainbow', 'hsv', 'flag', 'prism'])
        :param colormap: str of colormap
        """
        self.colormap = colormap

    def set_image_path(self, path):
        self.image_location = path

    def read_image(self, path_to_image=None):

        if path_to_image is None:
            self.image_location, pn = QtGui.QFileDialog.getOpenFileName(self,
                                                                        "Import Film2Dose files",
                                                                        QtCore.QDir.currentPath(),
                                                                        "Film2Dose Dose image (*.ftd)")

        else:
            self.image_location = path_to_image

        if self.image_location:
            QtCore.QDir.setCurrent(self.image_location)
            _, filepart = os.path.splitext(self.image_location)
            if self.image_location:
                data = load(self.image_location)
                self.im, self.delta, self.image_type, self.calib_data = data.get_image()
                if self.image_type == 'ftd-dose':
                    self.image_title = self.image_type_names[0]
                    self.calc_method = data.calc_method

                    return True

        else:
            return False

    def show_image(self, fig=None, ax=None):

        if fig is None and ax is None:
            # print(self.im.shape)
            im = self.im[:, :, self.channel]
            self.min_sat = np.percentile(im, 1)
            self.max_sat = np.percentile(im, 99)

            mi = str(round(self.min_sat))
            mx = str(round(self.max_sat))
            self.minLineEdit.setText(mi)
            self.maxLineEdit.setText(mx)
            lim = (self.min_sat, self.max_sat)
            self.fig, self.ax = display_fig(im, self.delta, self.colormap, lim, self.isocenter, self.scale)
        elif fig == 1:
            im = self.im[:, :, self.channel]
            lim = (self.min_sat, self.max_sat)
            print('limit:', lim)
            self.fig, self.ax = display_fig(im, self.delta, self.colormap, lim, self.isocenter, self.scale)
        else:
            self.fig = fig
            self.ax = ax

        if self.image_type == 'tif':
            title = self.image_type_names[-1]
        else:
            title = self.image_type_names[self.channel]

        self.ax.set_title(title + " - " + self.channel_names[self.channel])
        self.canvas = RotationCanvas(self.fig)
        self.verticalLayout_2.addWidget(self.canvas)
        self.navigation_toolbar = Film2DoseToolbar(self.canvas, self)
        self.navigation_toolbar.setIconSize(QtCore.QSize(46, 46))
        self.verticalLayout_2.addWidget(self.navigation_toolbar)

    def on_color(self, txt):
        self.colormap = txt
        self.update_image()

    def on_min(self):
        print('on_min')
        self.min_sat = float(self.minLineEdit.text())
        self.update_image(fig=1)

    def on_max(self):
        print('on_max')
        self.max_sat = float(self.maxLineEdit.text())
        self.update_image(fig=1)

    def get_canvas_points(self, n):
        self.update_image()
        self.cursor = Cursor(self.ax, useblit=True, color='red', linewidth=1)
        self.canvas.set_points(n)
        tmp = self.canvas.get_points()
        self.cursor.disconnect_events()
        pos = np.asarray(tmp[0])
        return pos

    def on_flipud(self):
        self.im = np.flipud(self.im)
        self.update_image()

    def on_fliplr(self):
        self.im = np.fliplr(self.im)
        self.update_image()

    def set_isocenter(self):
        self.update_image()
        self.cursor = Cursor(self.ax, useblit=True, color='red', linewidth=1)
        self.canvas.set_points(1)
        position = self.canvas.get_points()
        self.cursor_position = np.asarray(position[0]).astype(int)
        self.isocenter += np.asarray(position[0])
        print('Position: ' + str(self.cursor_position))
        print('actual isocenter: ' + str(self.isocenter))
        self.update_image()

    def save_images(self):

        h0, h1 = self.ax.get_xlim()
        v0, v1 = self.ax.get_ylim()
        imc = get_crop(self.im, self.delta, [h0, h1, v0, v1])
        print('limits: xlim: %s, %s  ylim: %s, %s' % (h0, h1, v0, v1))
        # TODO refactor dicomRT export
        im = Fim2DoseImage(imc, self.delta, self.image_type, self.isocenter, self.calib_data)
        file_name, _ = QtGui.QFileDialog.getSaveFileName(None, "Save Film2Dose image",
                                                         QtCore.QDir.currentPath(),
                                                         "Film2Dose images (*.ftd);; DICOM Dose (*.dcm)")

        im.set_calculation_method(self.calc_method)

        if file_name[-3:] == 'ftd':
            save_ftd(im, file_name)
        elif file_name[-3:] == 'dcm':
            save_dicom_dose(self.im[:, :, self.channel], self.delta, file_name)

    def get_image(self):
        return self.im[:, :, self.channel], self.delta

    def on_rotateCW(self):
        self.im = np.rot90(self.im, 3)
        self.update_image()

    def on_rotateCCW(self):
        self.im = np.rot90(self.im)
        self.update_image()

    def on_rotate(self):
        self.cursor = Cursor(self.ax, useblit=True, color='red', linewidth=1)
        position = self.canvas.get_points()
        x = (position[0][0], position[1][0])
        y = (position[0][1], position[1][1])
        self.im = rotate_image(self.im, x, y)
        self.update_image()

    def update_image(self, fig=None, ax=None):
        try:
            self.verticalLayout_2.removeWidget(self.canvas)
            self.canvas.setParent(None)
            self.verticalLayout_2.removeWidget(self.navigation_toolbar)
            self.navigation_toolbar.setParent(None)
            self.show_image(fig, ax)
        except:
            pass

    def window_adjust(self, channel):
        self.min_sat = np.percentile(self.im[:, :, channel], 1)
        self.max_sat = np.percentile(self.im[:, :, channel], 99)

    def on_activated(self):
        if self.channel_box.currentIndex() == 0:
            self.window_adjust(0)
            self.update_combo()
        elif self.channel_box.currentIndex() == 1:
            self.window_adjust(1)
            self.update_combo()
        elif self.channel_box.currentIndex() == 2:
            self.window_adjust(2)
            self.update_combo()

    def update_combo(self):
        self.channel = self.channel_box.currentIndex()
        try:
            self.verticalLayout_2.removeWidget(self.canvas)
            self.canvas.setParent(None)
            self.verticalLayout_2.removeWidget(self.navigation_toolbar)
            self.navigation_toolbar.setParent(None)
            self.show_image()
        except:
            pass


class Film2DoseToolbar(NavigationToolbar2):
    def _init_toolbar(self):
        super(Film2DoseToolbar, self)._init_toolbar()

    def __init__(self, canvas, parent, coordinates=True):
        super(Film2DoseToolbar, self).__init__(canvas, parent, coordinates)
        self.rect = None
        self.points = []
        self.crop_points = []
        self.crop_index = []

    def home(self, *args):
        super(Film2DoseToolbar, self).home(*args)
        self.crop_points = []

    def draw_rubberband(self, event, x0, y0, x1, y1):
        # its img[y: y + h, x: x + w]
        height = self.canvas.figure.bbox.height
        y1 = height - y1
        y0 = height - y0

        w = abs(x1 - x0)
        h = abs(y1 - y0)

        self.rect = [int(val) for val in (min(x0, x1), min(y0, y1), w, h)]
        self.canvas.drawRectangle(self.rect)
        self.points.append([event.xdata, event.ydata])

    def release_zoom(self, event):
        super(Film2DoseToolbar, self).release_zoom(event)

        if self.points:
            x0, y0 = self.points[0][0], self.points[0][1]
            x1, y1 = self.points[-1][0], self.points[-1][1]
            self.crop_points.append([x0, y0, x1, y1])
            self.points = []

            self.get_crop_points()

    def get_crop_points(self):

        if self.crop_points:
            return self.crop_points[-1]


class DoseConversionWidget(QtGui.QWidget, DoseConversionQT.Ui_Form_film2dose):
    def __init__(self, parent=None):
        super(DoseConversionWidget, self).__init__(parent)
        self.setupUi(self)
        self.file_name = ''
        self.cal = None  # calibration object
        self.calib_data = {}  # Calibration data
        self.fig = None
        self.channel = 0
        self.eq = 0
        self.crop_index = []
        self.image_widget = EditImageWidget()
        self.dose = None
        self.dose_widget = None
        self.method_titles = {'Single Channel': 'Single_Channel', 'Robust Average': 'Robust_Average',
                              'Robust Multichannel': 'Robust_RGB'}
        self.method = 'Single_Channel'
        self.setWindowTitle(
            QtGui.QApplication.translate("Film2Dose", "Dose Conversion", None, QtGui.QApplication.UnicodeUTF8))

        self.set_connections()
        self.set_param()

    def set_param(self):
        self.file_name, _ = QtGui.QFileDialog.getOpenFileName(self, "Open calibration object",
                                                              QtCore.QDir.currentPath(),
                                                              "Film2Dose calibration object (*.fco)")

        self.cal = load(self.file_name)
        self.calib_data['doses'] = self.cal.doses
        self.calib_data['cal_od'] = self.cal.cal_od
        self.calib_data['eqt'] = self.cal.calib_data['eqt']
        self.calib_data['sigparam'] = self.cal.sigparam
        self.image_widget.read_image()
        self.image_widget.show_image()
        self.gridLayout_2.addWidget(self.image_widget, 3, 1, 1, 1)

    def set_connections(self):
        self.to_dose.clicked.connect(self.dose_conversion)
        self.method_combo.activated[str].connect(self.on_method)
        self.calib_button.clicked.connect(self.on_calib)
        self.import_image.clicked.connect(self.on_import)

    def _on_import(self):
        self.gridLayout_2.removeWidget(self.dose_widget)
        self.gridLayout_2.removeWidget(self.image_widget)
        self.dose_widget = OptimizedDoseWidget()
        self.image_widget = EditImageWidget()
        self.image_widget.read_image()
        self.image_widget.show_image()
        self.gridLayout_2.addWidget(self.image_widget, 3, 1, 1, 1)

    def on_calib(self):
        self.file_name, _ = QtGui.QFileDialog.getOpenFileName(self, "Open calibration object",
                                                              QtCore.QDir.currentPath(),
                                                              "Film2Dose calibration object (*.fco)")

        self.cal = load(self.file_name)
        self.calib_data['doses'] = self.cal.doses
        self.calib_data['cal_od'] = self.cal.cal_od
        self.calib_data['eqt'] = self.cal.calib_data['eqt']
        self.calib_data['sigparam'] = self.cal.sigparam

    def on_method(self, txt):
        self.method = self.method_titles[txt]

    def dose_conversion(self):
        self.dose = Model(self.cal, od2pixel(self.image_widget.im), self.image_widget.delta)
        lat_corr = self.lateral_checkbox.isChecked()
        delta = self.image_widget.delta
        self.gridLayout_2.removeWidget(self.dose_widget)
        self.dose_widget = OptimizedDoseWidget()
        if self.method == 'Single_Channel':
            self.dose_widget.set_method(self.method)
            sc_dose = self.dose.single_channel_dose(lat_corr=lat_corr)
            self.dose_widget.set_image(sc_dose, delta, 0, 'ftd-dose', self.calib_data)
            self.dose_widget.show_image()
            self.gridLayout_2.addWidget(self.dose_widget, 3, 2, 1, 1)
        elif self.method == 'Robust_Average':
            self.dose_widget.set_method(self.method)
            rob_dose = self.dose.robust_average_dose(lat_corr=lat_corr)
            self.dose_widget.set_image(rob_dose, delta, 0, 'ftd-dose', self.calib_data)
            self.dose_widget.show_image()
            self.gridLayout_2.addWidget(self.dose_widget, 3, 2, 1, 1)
        elif self.method == 'Robust_RGB':
            self.dose_widget.set_method(self.method)
            rob_dose = self.dose.robust2dose(lat_corr=lat_corr)
            self.dose_widget.set_image(rob_dose, delta, 0, 'ftd-dose', self.calib_data)
            self.dose_widget.show_image()
            self.gridLayout_2.addWidget(self.dose_widget, 3, 2, 1, 1)

    def update_cropped(self):
        crop_position = self.toolbar.get_crop_points()
        if crop_position:
            self.crop_index = get_crop(self.im, self.delta, crop_position)

    def update_saturation(self):
        self.update_cropped()

    def on_import(self):
        QtGui.QMessageBox.information(None, "Warning", "You already have an image!")

        reply = QtGui.QMessageBox.question(self, "Warning",
                                           "You already have an image! Do you want to import a different file?",
                                           QtGui.QMessageBox.Yes | QtGui.QMessageBox.No | QtGui.QMessageBox.Cancel)

        if reply == QtGui.QMessageBox.Yes:
            self._on_import()
        elif reply == QtGui.QMessageBox.No:
            self.questionLabel.setText("No")
        else:
            self.questionLabel.setText("Cancel")


class BatchDoseConversionWidget(DoseConversionWidget):
    def __init__(self, parent=None):
        self.image_files = []
        self.pbar = None
        super(BatchDoseConversionWidget, self).__init__(parent)
        self.setWindowTitle(
            QtGui.QApplication.translate("Dose Conversion", "Batch Dose Conversion", None,
                                         QtGui.QApplication.UnicodeUTF8))

        self.resize(400, 400)

    def set_param(self):
        self.file_name, _ = QtGui.QFileDialog.getOpenFileName(self, "Open calibration object",
                                                              QtCore.QDir.currentPath(),
                                                              "Film2Dose calibration object (*.fco)")

        self.image_files, _ = QtGui.QFileDialog.getOpenFileNames(self, "Open 48 bits tiff Calibration Files",
                                                                 QtCore.QDir.currentPath(),
                                                                 "48 bit Tiff Files (*.tif)")

        self.cal = load(self.file_name)
        self.calib_data['doses'] = self.cal.doses
        self.calib_data['cal_od'] = self.cal.cal_od
        self.calib_data['eqt'] = self.cal.calib_data['eqt']
        self.calib_data['sigparam'] = self.cal.sigparam

    def dose_conversion(self):
        lat_corr = self.lateral_checkbox.isChecked()

        if self.pbar is not None:
            self.pbar.close()

        self.pbar = QtGui.QProgressBar(self)
        if self.method == 'Single_Channel':

            self.pbar.move(150, 150)
            self.setWindowTitle('Optimizing')
            self.pbar.setMinimum(0)
            self.pbar.setMaximum(len(self.image_files))
            self.pbar.show()
            step = 0

            for path in self.image_files:
                # direct dose conversion
                self.image_widget.read_image(path)
                self.dose = Model(self.cal, od2pixel(self.image_widget.im), self.image_widget.delta)
                sc_dose = self.dose.single_channel_dose(lat_corr=lat_corr)
                step += 1
                self.pbar.setValue(step)
                im = Fim2DoseImage(sc_dose, self.dose.delta, 'ftd-dose', calib_data=self.calib_data)
                im.set_calculation_method(self.method)
                fileName, fileExtension = os.path.splitext(path)
                file_name = fileName + '-dose.ftd'
                save(im, file_name)

        if self.method == 'Robust_Average':
            self.pbar.move(150, 150)
            self.setWindowTitle('Optimizing')
            self.pbar.setMinimum(0)
            self.pbar.setMaximum(len(self.image_files))
            self.pbar.show()
            step = 0

            for path in self.image_files:
                # direct dose conversion
                self.image_widget.read_image(path)
                self.dose = Model(self.cal, od2pixel(self.image_widget.im), self.image_widget.delta)
                sc_dose = self.dose.robust_average_dose(lat_corr=lat_corr)
                step += 1
                self.pbar.setValue(step)
                im = Fim2DoseImage(sc_dose, self.dose.delta, 'ftd-dose', calib_data=self.calib_data)
                im.set_calculation_method(self.method)
                fileName, fileExtension = os.path.splitext(path)
                file_name = fileName + '-dose.ftd'
                save(im, file_name)

        if self.method == 'Robust_RGB':
            self.pbar.move(150, 150)
            self.setWindowTitle('Optimizing')
            self.pbar.setMinimum(0)
            self.pbar.setMaximum(len(self.image_files))
            self.pbar.show()
            step = 0

            for path in self.image_files:
                # direct dose conversion
                self.image_widget.read_image(path)
                self.dose = Model(self.cal, od2pixel(self.image_widget.im), self.image_widget.delta)
                sc_dose = self.dose.robust2dose(lat_corr=lat_corr)
                step += 1
                self.pbar.setValue(step)
                im = Fim2DoseImage(sc_dose, self.dose.delta, 'ftd-dose', calib_data=self.calib_data)
                im.set_calculation_method(self.method)
                fileName, fileExtension = os.path.splitext(path)
                file_name = fileName + '-dose.ftd'
                save(im, file_name)


# noinspection PyUnresolvedReferences
class FitWidget(QtGui.QWidget, FitCurvesQt.Ui_Form):
    fit_done = QtCore.Signal(object)

    def __init__(self, parent=None):
        super(FitWidget, self).__init__(parent)
        self.file_name = ''
        self.cal = None
        self.fig = None
        self.channel = 0
        self.eq = 0
        self.setupUi(self)
        self.set_connections()

    def set_cal(self, cal):
        self.cal = cal

    def set_connections(self):

        self.comboBox.activated.connect(self.on_activated)
        self.radioButton_eq1.clicked.connect(self.on_radio1)
        self.RadioButton_eq2.clicked.connect(self.on_radio2)
        self.radioButton_3.clicked.connect(self.on_radio3)
        self.select_curve.clicked.connect(self.curve_select)
        self.finish_button.clicked.connect(self.save_calib)

    def on_radio1(self):
        self.update()
        self.eq = 1
        self.calib_channel()
        self._show_image()

    def on_radio2(self):
        self.update()
        self.eq = 2
        self.calib_channel()
        self._show_image()

    def on_radio3(self):
        self.update()
        self.eq = 3
        self.calib_channel()
        self._show_image()

    def set_param(self):
        self.file_name, _ = QtGui.QFileDialog.getOpenFileName(self, "Open calibration object",
                                                              QtCore.QDir.currentPath(),
                                                              "Film2Dose calibration object (*.fco)")
        self.cal = load(self.file_name)

    def on_activated(self):

        if self.comboBox.currentIndex() == 0:
            self.update_combo()
        elif self.comboBox.currentIndex() == 1:
            self.update_combo()
        elif self.comboBox.currentIndex() == 2:
            self.update_combo()

    def update_combo(self):
        self.channel = self.comboBox.currentIndex()
        self.update()
        self.eq = 1
        self.calib_channel()
        self._show_image()
        # self.radioButton_eq1.setChecked(True)

    def calib_channel(self):
        # print('Calibrating channel: ' + self.channels[channel])
        try:
            # self.cal.fit_curve(self.channel, self.eq)
            self.fig = self.cal.show_curve(self.channel, self.eq)

        except:
            msg = "Check your calibration data!"
            QtGui.QMessageBox.critical(self, "Incomplete calibration data",
                                       msg,
                                       QtGui.QMessageBox.Abort)

    def _show_image(self):
        """
            Shows a Matplotlib Canvas on a QT OptimizeCalibration windows using a Navigation Toolbar.

        """
        self.figure_canvas = FigureCanvas(self.fig)
        self.verticalLayout_8.addWidget(self.figure_canvas)
        self.navigation_toolbar = NavigationToolbar2(self.figure_canvas, self)
        self.navigation_toolbar.setIconSize(QtCore.QSize(46, 46))
        self.verticalLayout_8.addWidget(self.navigation_toolbar, 0)

    def curve_select(self):

        channel_txt = self.comboBox.currentText()

        if not all(self.cal.is_fitted.values()):
            msg = "<p> Are you selecting Equation %s to %s ?</p>" % (self.eq, channel_txt)
            reply = QtGui.QMessageBox.question(self, "Confirm your choice",
                                               msg,
                                               QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)
            if reply == QtGui.QMessageBox.Yes:
                self.fit_channel()
        else:
            cl = [key for key in self.cal.is_fitted.keys() if self.cal.is_fitted[key] is True]
            tx = ' and '.join(cl)
            message = "<p>There are selected equations for %s channel</p>" \
                      "<p>Do you want to select a different equations for each channel?</p>" % tx
            reply = QtGui.QMessageBox.question(self, "Confirm your choice",
                                               message,
                                               QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)
            if reply == QtGui.QMessageBox.Yes:
                self.cal.reset()
                self.fit_channel()

    def fit_channel(self):

        pol, sigma, R, df = self.cal.fit_curve(self.channel, self.eq)
        self.cal.rgb_pol.append(np.flipud(pol))  # inverting coeff order
        self.cal.fit_unc.append([sigma, R, df])
        self.cal.eqt.append(self.eq)
        self.cal.sigparam[self.channel] = self.cal.calc_uncertainty(self.channel, self.eq)
        self.cal.calib_data[self.cal.channels[self.channel]] = self.cal.get_interp(self.eq, pol)
        # setting channel calibrated
        self.cal.is_fitted[self.cal.channels[self.channel]] = True

    @QtCore.Slot()
    def save_calib(self):

        if all(self.cal.is_fitted.values()):
            file_name, _ = QtGui.QFileDialog.getSaveFileName(None, "Save calibration object",
                                                             QtCore.QDir.currentPath(),
                                                             "Film2Dose calibration object (*.fco)")

            self.cal.calib_rgb()
            self.fit_done.emit(self.cal)
            if file_name:
                save(self.cal, file_name)

        else:
            cl = [key for key in self.cal.is_fitted.keys() if self.cal.is_fitted[key] is False]
            tx = ' and '.join(cl)
            message = "<p>You still need to select equations for %s channel</p>" % tx
            QtGui.QMessageBox.information(None, "Missing Data", message)

    @property
    def fitted_cal(self):
        return self.cal

    def update(self):
        try:
            self.verticalLayout_8.removeWidget(self.figure_canvas)
            self.figure_canvas.setParent(None)
            self.verticalLayout_8.removeWidget(self.navigation_toolbar)
            self.navigation_toolbar.setParent(None)
        except:
            pass


class MplCalibrationWidget(QtGui.QWidget):
    def __init__(self, parent, fig, npoints=None):
        QtGui.QWidget.__init__(self, parent)
        self.canvas = MplCalibrationCanvas(fig)

        # THESE TWO LINES WERE ADDED
        self.canvas.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.canvas.setFocus()
        # # Add a Main layout

        self.l = QtGui.QVBoxLayout(parent)
        self.l.addWidget(self.canvas)
        self.navigation_toolbar = NavigationToolbar2(self.canvas, self)
        self.navigation_toolbar.setIconSize(QtCore.QSize(46, 46))
        self.l.addWidget(self.navigation_toolbar)

    def get_points(self):
        return self.canvas.get_points()

    def add_table(self, table):
        self.l.addWidget(table)


class CalibrationWidget(QtGui.QWidget):
    def __init__(self, parent=None):
        super(CalibrationWidget, self).__init__(parent)
        self.setWindowTitle(
            QtGui.QApplication.translate("Film2Dose", "gafchromic film calibration", None,
                                         QtGui.QApplication.UnicodeUTF8))

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/icons/scanner.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.setWindowIcon(icon)
        self.mpl_widget = None
        self.resize(1280, 800)
        self.cal = None
        self.tableWidget = QtGui.QTableWidget()
        self.load_cal_images()

    def load_cal_images(self):
        file_name, _ = QtGui.QFileDialog.getOpenFileNames(self, "Open 48 bits tiff Calibration Files",
                                                          QtCore.QDir.currentPath(),
                                                          "48 bit Tiff Files (*.tif)")
        actual_path = os.path.dirname(file_name[0])
        channel = 0  # red

        if file_name:
            self.cal = Film2DoseCalibration(filename=file_name)
            self.cal.read_image()
            file_name, _ = QtGui.QFileDialog.getOpenFileName(self, "Open calibration doses file",
                                                             actual_path,
                                                             "txt Files (*.txt)")

            doses = read_cal_doses(file_name)
            self.cal.set_dose_points(doses)
            cim = self.cal.calib_image[:, :, channel]
            calim = np.flipud(cim)
            # show image to calibrate
            fig, ax = display_fig(calim, self.cal.delta, col_map='Greys', limits=(0, 1))
            ax.set_title('Dose calibration points')

            # Attach to Widget
            self.mpl_widget = MplCalibrationWidget(self, fig)

            self.show()

            # get dose points Position
            position = self.mpl_widget.get_points()
            self.cal.set_points_position(position)

            # fill Table Widget
            self._fill_table()
            self.mpl_widget.add_table(self.tableWidget)
            file_name, _ = QtGui.QFileDialog.getSaveFileName(None, "Save calibration object",
                                                             QtCore.QDir.currentPath(),
                                                             "Film2Dose calibration object (*.fco)")
            if file_name:
                save(self.cal, file_name)

    def _fill_table(self):

        n = len(self.cal.doses)
        self.tableWidget.setRowCount(n)
        self.tableWidget.setColumnCount(4)
        labels = ['Dose (cGy)', 'OD (red)', 'OD (green)', 'OD (blue)']
        self.tableWidget.setHorizontalHeaderLabels(labels)

        for row in range(n):
            self.tableWidget.setItem(row, 0, QtGui.QTableWidgetItem(str("%s" % self.cal.doses[row])))
            self.tableWidget.setItem(row, 1, QtGui.QTableWidgetItem(str("%s" % self.cal.cal_od[row, 0])))
            self.tableWidget.setItem(row, 2, QtGui.QTableWidgetItem(str("%s" % self.cal.cal_od[row, 1])))
            self.tableWidget.setItem(row, 3, QtGui.QTableWidgetItem(str("%s" % self.cal.cal_od[row, 2])))


class WorkerThread(QtCore.QThread):
    def __init__(self, parent=None):
        super(WorkerThread, self).__init__(parent)
        self.dose = None

    def set_parameters(self, dose):
        self.dose = dose

    def run(self):
        self.dose.robust2dose(lat_corr=True)
        self.emit(QtCore.SIGNAL('threadDone(QString)'), 'Confirmation that the thread is finished.')
        print('Done whith the tread.')

    def get_calc_dose(self):
        return self.dose


class GridEditDialog(QtGui.QDialog, EditGridQT.Ui_Dialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.grid_data = np.array([])
        self.set_connections()

    def set_connections(self):
        self.buttonBox.accepted.connect(self.on_accept)
        self.buttonBox.rejected.connect(self.on_reject)

    def on_accept(self):
        nx = self.nx_spin.value()
        x_delta = self.xd_spin.value()
        ny = self.ny_spin.value()
        ny_delta = self.yd_spin.value()
        self.grid_data = np.array([nx, x_delta, ny, ny_delta])

    def get_grid(self):
        if np.all(self.grid_data > 0):
            return self.grid_data

    def on_reject(self):
        pass


class GetDataWidget(QtGui.QWidget, GetCalPointsQT.Ui_GetCalPoints):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.method = 0  # 0: point, 1: grid data
        self.reference_widget = TPSWidget()
        self.film_od_widget = EditImageWidget()
        self.bg_data = EditImageWidget()
        self.grid_dialog = None
        self.cal_obj = None
        self.is_reg = False
        self.ref_resc = None
        self.film_resc = None
        self.bg_resc = None
        self.grid_data = []
        self.cal_doses = []
        self.cal_od = []
        self.delta_resc = 1
        self.table_widget = QtGui.QTableWidget()
        self.result_widget = GetDataResultWidget()
        self.set_connections()

    def set_connections(self):
        self.load_data_button.clicked.connect(self.on_load)
        self.get_points_Button.clicked.connect(self.on_get_points)
        self.save_button.clicked.connect(self.on_save)
        self.point_radio.clicked.connect(self.on_point)
        self.grid_radio.clicked.connect(self.on_grid)
        self.register_button.clicked.connect(self.on_register)

    def on_load(self):
        QtGui.QMessageBox.information(None, "TPS", 'Import TPS reference doses (cGy)')
        self.reference_widget.read_image()
        self.reference_widget.show_image()
        QtGui.QMessageBox.information(None, "Film", 'Import film irradiated doses (cGy)')
        self.film_od_widget.read_image()
        self.film_od_widget.show_image()
        QtGui.QMessageBox.information(None, "Film", 'Import non irradiated film (BG)')
        self.bg_data.read_image()
        self.gridLayout_2.addWidget(self.reference_widget, 6, 1, 1, 1)
        self.gridLayout_2.addWidget(self.film_od_widget, 6, 2, 1, 1)

    def on_register(self):
        # TODO implement image manipulation actions before registration, then apply in BG image the same actions.
        if not self.is_reg:
            ref_im, ref_delta = self.reference_widget.get_image()
            raw_od, cal_delta = self.film_od_widget.get_all_channels()
            bg_im, _ = self.bg_data.get_all_channels()
            self.cal_obj = Film2DoseCalibration(ref_im, ref_delta, raw_od, cal_delta, bg_im)
            self.ref_resc, self.film_resc, self.bg_resc, self.delta_resc = self.cal_obj.rescale_images(trim=5)
            self.reference_widget.set_image(self.ref_resc, self.delta_resc)
            self.reference_widget.update_image()
            self.film_od_widget.set_image(pixel2od(self.film_resc), self.delta_resc)
            self.film_od_widget.update_image()
            self.is_reg = True
            self.cal_obj.set_registered(self.is_reg)
        else:
            QtGui.QMessageBox.information(None, "Information", 'Images already registered')

    def on_get_points(self):
        if self.cal_obj is not None:
            if self.method:
                # Grid method
                self.grid_dialog = GridEditDialog()
                self.grid_dialog.exec_()
                self.grid_data = self.grid_dialog.get_grid()
                pos = self.film_od_widget.get_position()
                # select upper left point of grid (0,0)
                self.cal_obj.set_point_position(pos)
                self.cal_doses, self.cal_od = self.cal_obj.auto_get_points(grid=self.grid_data, roi_size=(5, 5))
                self._fill_table()
                self.gridLayout_2.addWidget(self.table_widget, 7, 1, 1, 1)
                fig, ax = plot_cal_data(self.cal_doses, self.cal_od)
                self.result_widget.set_figure(fig)
                self.result_widget.show_figure()
                self.gridLayout_2.addWidget(self.result_widget, 7, 2, 1, 1)
            else:
                # Point method
                npoints, ok = QtGui.QInputDialog.getInteger(None, "Enter...", "Number of calibration points")
                pos = self.reference_widget.get_points(npoints)
                self.cal_doses, self.cal_od = self.cal_obj.get_calib_data(pos)
                self._fill_table()
                self.gridLayout_2.addWidget(self.table_widget, 7, 1, 1, 1)
                fig, ax = plot_cal_data(self.cal_doses, self.cal_od)
                self.result_widget.set_figure(fig)
                self.result_widget.show_figure()
                self.gridLayout_2.addWidget(self.result_widget, 7, 2, 1, 1)
        else:
            QtGui.QMessageBox.information(None, "Information", 'You need to register images first!')

    def _fill_table(self):

        n = len(self.cal_doses)
        self.table_widget.setRowCount(n)
        self.table_widget.setColumnCount(4)
        labels = ['Dose (cGy)', 'OD (red)', 'OD (green)', 'OD (blue)']
        self.table_widget.setHorizontalHeaderLabels(labels)

        for row in range(n):
            self.table_widget.setItem(row, 0, QtGui.QTableWidgetItem(str("%s" % self.cal_doses[row])))
            self.table_widget.setItem(row, 1, QtGui.QTableWidgetItem(str("%s" % self.cal_od[row, 0])))
            self.table_widget.setItem(row, 2, QtGui.QTableWidgetItem(str("%s" % self.cal_od[row, 1])))
            self.table_widget.setItem(row, 3, QtGui.QTableWidgetItem(str("%s" % self.cal_od[row, 2])))

    def on_save(self):

        if np.any(self.cal_od):
            file_name, _ = QtGui.QFileDialog.getSaveFileName(None, "Save calibration object",
                                                             QtCore.QDir.currentPath(),
                                                             "Film2Dose calibration object (*.fco)")
            if file_name:
                save(self.cal_obj, file_name)
        else:
            QtGui.QMessageBox.information(None, "Information", 'You need to get all calibration data first!')

    def on_point(self):
        self.method = 0

    def on_grid(self):
        self.method = 1


class GetDataResultWidget(QtGui.QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fig = None
        self.canvas = None
        self.navigation_toolbar = None
        self.verticalLayout = QtGui.QVBoxLayout(self)
        self.setLayout(self.verticalLayout)

    def set_figure(self, fig):
        self.fig = fig

    def show_figure(self):
        self.canvas = FigureCanvas(self.fig)
        FigureCanvas.setSizePolicy(self, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.verticalLayout.addWidget(self.canvas)
        self.navigation_toolbar = Film2DoseToolbar(self.canvas, self)
        self.navigation_toolbar.setIconSize(QtCore.QSize(46, 46))
        self.verticalLayout.addWidget(self.navigation_toolbar)


class CalibrationModeWidget(QtGui.QWidget, FitModeQT.Ui_FitMode):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.cal_obj = None
        self.fit_widget = FitWidget()
        self.evo_widget = OptimizationWidget()
        self.set_connections()

    def set_connections(self):
        self.load_button.clicked.connect(self.on_load_data)
        self.fit_channels_button.clicked.connect(self.on_fit_channels)
        self.save_cal_button.clicked.connect(self.on_save)
        self.evo_button.clicked.connect(self.on_evo)
        self.fit_widget.raise_()
        self.fit_widget.fit_done.connect(self.fit_done)

    def on_load_data(self):
        file_name, _ = QtGui.QFileDialog.getOpenFileName(None, "Open calibration object",
                                                         QtCore.QDir.currentPath(),
                                                         "Film2Dose calibration object (*.fco)")
        self.cal_obj = load(file_name)

    def on_fit_channels(self):
        if self.cal_obj is not None:
            self.fit_widget.set_cal(self.cal_obj)
            self.fit_widget.show()
            self.fit_widget.raise_()
        else:
            QtGui.QMessageBox.information(None, "Information", 'You need to import the calibration object (*.fco)')

    @QtCore.Slot(object)
    def fit_done(self, message):

        self.cal_obj = message
        print('received: ' + message.__str__())

    def on_save(self):
        if self.cal_obj is not None:
            if all(self.cal_obj.is_fitted.values):
                file_name, _ = QtGui.QFileDialog.getSaveFileName(None, "Save calibration object",
                                                                 QtCore.QDir.currentPath(),
                                                                 "Film2Dose calibration object (*.fco)")
                if file_name:
                    self.cal.calib_rgb()
                    save(self.cal, file_name)

            else:
                cl = [key for key in self.cal_obj.is_fitted.keys() if self.cal.is_fitted[key] is False]
                tx = ' and '.join(cl)
                message = "<p>You still need to select equations for %s channel</p>" % tx
                QtGui.QMessageBox.information(None, "Missing Data", message)

    def on_evo(self):
        if self.cal_obj is not None:
            if self.cal_obj.is_reg:
                self.evo_widget.set_calibration(self.cal_obj)

                self.evo_widget.show()
            else:
                # todo add reg option
                QtGui.QMessageBox.information(None, "Information", 'You have to register images first')
        else:
            QtGui.QMessageBox.information(None, "Information", 'You need to import the calibration object (*.fco)')


class OptimizationWidget(QtGui.QWidget, OptimizationQT.Ui_Form):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.cal_obj = Film2DoseCalibration()
        self.eq = 1
        self.eq_name = 'Equation 1 - Inverse Log poly'
        self.channel = 0  # red color
        self.mode = 'poly'
        self.evo_method = {1: 'ES', 2: 'GA'}
        self.method = 'ES'
        self.color_channels = {'Red': 0, 'Green': 1, 'Blue': 2}
        self.channel_name = 'Red'
        self.eq_titles = {'Equation 1 - Inverse Log poly': 1, 'Equation 2 - Inverse poly': 2,
                          'Equation 3 - Inverse arctan poly': 3, 'Equation 4 - 4th Degree Poly': 4}

        self.eq_name = ''
        self.mode_titles = {'Polynomial curve fitting': 'poly', 'Poly fit and correction': 'polylateral',
                            'Lateral correction': 'lateral'}
        self.is_netod = False
        self.poly_param = None
        self.is_set = False
        self.ref_widget = TPSWidget()
        self.image_widget = EditImageWidget()
        self.set_connections()

    def set_connections(self):
        self.setup_button.clicked.connect(self.on_setup)
        self.optimize_button.clicked.connect(self.on_optimize)
        self.color_combo.activated[str].connect(self.on_color)
        self.mode_combo.activated[str].connect(self.on_mode)
        self.eq_combo.activated[str].connect(self.on_equation)
        self.save_cal.clicked.connect(self.save_calib)

    def on_mode(self, txt):
        self.mode = self.mode_titles[txt]

    def on_color(self, txt):
        self.channel = self.color_channels[txt]
        self.channel_name = txt

    def on_equation(self, txt):
        self.eq = self.eq_titles[txt]
        self.eq_name = txt

    def set_calibration(self, cal):
        self.cal_obj = cal
        self._show_images()

    def _show_images(self):
        if not self.cal_obj.is_reg:
            ref, ref_delta, film, film_delta = self.cal_obj.get_cal_images()
            self.pixel_size_spin.setValue(ref_delta)
            self.ref_widget.set_image(ref, ref_delta)
            self.ref_widget.show_image()
            self.image_widget.set_image(self.film, film_delta)
            self.image_widget.show_image()
            self.gridLayout.addWidget(self.image_widget, 10, 1, 1, 1)
            self.gridLayout.addWidget(self.ref_widget, 10, 0, 1, 1)
            QtGui.QMessageBox.information(None, "Information",
                                          'You have to register images match images orientation before optimization.')
        else:
            ref_resc, film_resc, delta_resc = self.cal_obj.get_reg_images()
            self.ref_widget.set_image(ref_resc, delta_resc)
            self.ref_widget.show_image()
            self.image_widget.set_image(pixel2od(film_resc), delta_resc)
            self.image_widget.show_image()
            self.gridLayout.addWidget(self.image_widget, 10, 1, 1, 1)
            self.gridLayout.addWidget(self.ref_widget, 10, 0, 1, 1)

    def _rescale_images(self):
        pix_mm = self.pixel_size_spin.value()
        trim_mm = self.crop_border_spin.value()
        ref, ref_delta, film, film_delta = self.cal_obj.get_cal_images()
        ref_im, ref_delta = dowsampling_image(ref, ref_delta, pix_mm)
        self.cal_obj.set_reference(ref_im, ref_delta)
        ref_resc, film_resc, bg_resc, delta_resc = self.cal_obj.rescale_images(trim=trim_mm)
        self.ref_widget.set_image(ref_resc, delta_resc)
        self.ref_widget.update_image()
        self.image_widget.set_image(pixel2od(film_resc), delta_resc)
        self.image_widget.update_image()

    def on_setup(self):
        self._rescale_images()
        self.is_netod = self.bg_checkBox.isChecked()
        poly_bounds = self.poly_range_spin.value()
        pop_size = self.pop_spin.value()
        if self.mode == 'lateral' and not self.is_netod:
            if all(self.cal_obj.is_fitted.values):
                data = self.cal_obj.get_calibration()
                self.poly_param = np.ravel(data['poly'][self.channel])

        elif self.mode == 'lateral' and self.is_netod:
            data = self.cal_obj.get_calibration()
            self.poly_param = data['nod_data']['poly'][self.channel_name]['poly_optim']

        self.cal_obj.setup_optimization(eq=self.eq, ch=self.channel, mode=self.mode, poly_param=self.poly_param,
                                        net_od=self.is_netod, poly_bounds=poly_bounds, pop_size=pop_size)
        self.is_set = True

    def on_optimize(self):

        if self.is_set:
            self.cal_obj.set_seed(self.seed_spin.value())
            QtGui.QMessageBox.information(None, 'Start Optimization', 'Optimizing')

            self.cal_obj.optimize(method=self.method, display=True, channel_name=self.channel_name, mode=self.mode,
                                  net_od=self.is_netod)
            self.cal_obj.is_optimized[self.mode][self.channel_name] = True
            msg = 'The channel: ' + self.channel_name + ' is optimized using mode : ' + self.mode
            QtGui.QMessageBox.information(None, 'Done!', msg)
            self.is_set = False
        else:
            QtGui.QMessageBox.information(None, "Information", 'You have to set up your optimization.')

    def save_calib(self):

        if all(self.cal_obj.is_optimized[self.mode].values):
            file_name, _ = QtGui.QFileDialog.getSaveFileName(None, "Save calibration object",
                                                             QtCore.QDir.currentPath(),
                                                             "Film2Dose calibration object (*.fco)")

            self.cal_obj.calib_rgb()
            if file_name:
                save(self.cal_obj, file_name)

        else:
            cl = [key for key in self.cal_obj.is_optimized.keys() if self.cal_obj.is_optimized[key] is False]
            tx = ' and '.join(cl)
            message = "<p>You still need to optimize %s channel</p>" % tx
            QtGui.QMessageBox.information(None, "Missing Data", message)


class OptimizationThread(QtCore.QThread):
    def __init__(self, parent=None):
        super(OptimizationThread, self).__init__(parent)
        self.cal_obj = None
        self.method = ''
        self.channel_name = ''

    def set_parameters(self, cal, method, channel_name):
        self.cal_obj = cal
        self.method = method
        self.channel_name = channel_name

    def run(self):
        self.cal_obj.optimize(method=self.method, display=True, channel_name=self.channel_name)
        self.cal_obj.is_optimized[self.channel_name] = True
        self.emit(QtCore.SIGNAL('threadDone(QString)'), 'Confirmation that the thread is finished.')
        print('Done whith the tread.')

    def get_optimized_channel(self):
        return self.cal_obj


class GammaThread(QtCore.QThread):
    def __init__(self, parent=None):
        super(GammaThread, self).__init__(parent)
        self.func = None
        self.computed = None
        self.film = None
        self.dta = None
        self.dd = None
        self.dt = None
        self.g = None
        self.local = None

    def set_parameters(self, func, computed, film, dta, dd, dt, local):
        self.func = func
        self.computed = computed
        self.film = film
        self.dta = dta
        self.dd = dd
        self.dt = dt
        self.local = local

    def run(self):
        time.sleep(0.05)
        self.g = self.func(self.computed, self.film, self.dta, self.dd, self.dt, self.local)
        self.emit(QtCore.SIGNAL('threadDone(QString)'), 'Confirmation that the thread is finished.')
        print('Done whith the tread.')

    def get_gamma_index(self):
        return self.g


class ui_licence(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("Film2Dose")
        MainWindow.resize(654, 467)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtGui.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        spacerItem = QtGui.QSpacerItem(507, 26, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 0, 1, 1, 1)
        self.textEdit = QtGui.QTextEdit(self.centralwidget)
        self.textEdit.setObjectName("textEdit")
        self.gridLayout.addWidget(self.textEdit, 1, 0, 1, 2)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 654, 29))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(
            QtGui.QApplication.translate("Film2Dose", "Film2Dose", None, QtGui.QApplication.UnicodeUTF8))


class LicenceWidget(QtGui.QMainWindow, ui_licence):
    def __init__(self, parent=None):
        super(LicenceWidget, self).__init__(parent)
        self.setupUi(self)
        self.file_read()

    def file_read(self):
        with open('licence.txt', encoding="utf8") as f:
            self.textEdit.setText(f.read())


class MplCalibrationCanvas(FigureCanvas, BlockingMouseInput):
    """Class to represent the FigureCanvas widget"""

    def __init__(self, fig):
        self.fig = fig
        self.cid = None
        self.ax = self.fig.add_subplot(111)
        self.npoints = 0
        self.clicks = []
        self.show_clicks = True
        self.marks = []
        self.cal_points = []
        FigureCanvas.__init__(self, self.fig)
        FigureCanvas.setSizePolicy(self, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.npoints, ok = QtGui.QInputDialog.getInteger(None,
                                                         "Enter...", "Number of calibration points")

    def _on_click(self, event):
        """
             Matplotlib event Handler function.
        :param event: Matplotlib event
        """
        if event.xdata is not None and event.ydata is not None:
            self.add_event(event)
            if event.button == 1:
                self.mouse_event_add(event)
            elif event.button == 3:
                self.mouse_event_pop(event)

            print(len(self.clicks))
            if len(self.clicks) == self.npoints:
                self.fig.canvas.mpl_disconnect(self.cid)

    def get_points(self, timeout=1000000, show_clicks=True, mouse_add=1,
                   mouse_pop=3, mouse_stop=2):

        blocking_mouse_input = Film2DoseBlockingMouseInput(self.fig,
                                                           mouse_add=mouse_add,
                                                           mouse_pop=mouse_pop,
                                                           mouse_stop=mouse_stop)
        return blocking_mouse_input(n=self.npoints, timeout=timeout,
                                    show_clicks=show_clicks)


class RotationCanvas(MplCalibrationCanvas):
    def __init__(self, fig):
        self.fig = fig
        self.cid = None
        self.ax = self.fig.add_subplot(111)
        self.npoints = 2
        self.clicks = []
        self.show_clicks = True
        self.marks = []
        self.cal_points = []
        FigureCanvas.__init__(self, self.fig)
        FigureCanvas.setSizePolicy(self, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def set_points(self, n):
        self.npoints = n


class FieldWidgetTool(QtGui.QWidget, FieldWidgetQT.Ui_Form):
    def __init__(self, parent=None):
        super(FieldWidgetTool, self).__init__(parent)
        self.setupUi(self)
        self.image_location = ''
        self.im = None
        self.delta = None
        self.field_widget = None
        self.result = None
        self.field_profile = None
        self.set_connections()

    def set_connections(self):
        self.open_button.clicked.connect(self.on_open)
        self.analyse_button.clicked.connect(self.on_analyse)

    def show_widget(self):
        self.field_widget.show_image()
        self.gridLayout.addWidget(self.field_widget, 1, 0, 1, 3)

    def on_open(self):
        self.image_location, pn = QtGui.QFileDialog.getOpenFileName(self,
                                                                    "Import 48 bits tiff File or Film2Dose image files.",
                                                                    QtCore.QDir.currentPath(),
                                                                    "Film2Dose Dose images (*.ftd);;"
                                                                    "Tiff Files (*.tif);;"
                                                                    "Film2Dose images (*.fti);;"
                                                                    "DICOM Images (*.dcm)")

        QtCore.QDir.setCurrent(self.image_location)

        _, filepart = os.path.splitext(self.image_location)

        if self.image_location:
            if filepart in ['.tif', '.fti']:
                self.field_widget = EditImageWidget()
                self.field_widget.read_image(self.image_location)
                self.show_widget()

            elif filepart == '.ftd':
                self.field_widget = OptimizedDoseWidget()
                self.field_widget.read_image(self.image_location)
                self.show_widget()
            elif filepart in ['.dcm', '.flu', '.dat', '.35MM1', '.1', '.All', '.txt', '.opg']:
                self.field_widget = TPSWidget()
                self.field_widget.read_image(self.image_location)
                self.show_widget()
            else:
                QtGui.QMessageBox.information(None, "Error!", "Wrong image format")

    def on_analyse(self):
        im, delta = self.field_widget.get_image()
        self.result = SymmetryFlatness(im, delta)
        plt.show()


class FieldProfileWidget(QtGui.QWidget):
    def __init__(self, *args, **kwargs):
        super(FieldProfileWidget, self).__init__(*args, **kwargs)

        self.fig = None
        self.canvas = None
        self.navigation_toolbar = None
        self.setWindowTitle(
            QtGui.QApplication.translate("Film2Dose", "Symmetry and flatness results", None,
                                         QtGui.QApplication.UnicodeUTF8))
        self.verticalLayout = QtGui.QVBoxLayout(self)
        self.setLayout(self.verticalLayout)
        self.obj = None

    def set_obj(self, obj):
        self.obj = obj

    def show_figure(self):
        self.canvas = FigureCanvas(self.obj.fig)
        FigureCanvas.setSizePolicy(self, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.verticalLayout.addWidget(self.canvas)
        self.navigation_toolbar = Film2DoseToolbar(self.canvas, self)
        self.navigation_toolbar.setIconSize(QtCore.QSize(46, 46))
        self.verticalLayout.addWidget(self.navigation_toolbar)
        self.canvas.mpl_connect('button_press_event', self.obj.click)
        self.obj.reset_button.on_clicked(self.obj.clear_xy_subplots)
        self.canvas.draw()


# TODO IMPLEMENTAR ANALISE DE CAMPOS PEQUENOS

class StarShotWidget(QtGui.QWidget, StarShotQT.Ui_StarShotWidget):
    def __init__(self, parent=None):
        super(StarShotWidget, self).__init__(parent)
        self.setupUi(self)
        self.image_widget = EditImageWidget()
        self.image_widget.read_image()
        self.image_widget.set_colormap('jet')
        self.image_widget.show_image()
        self.star_obj = StarShot()
        self.verticalLayout.addWidget(self.image_widget)
        self.result_widget = StarShotResultWidget()
        self.mode = {1: 'auto', 0: 'manual'}
        self.mode_select = 1
        self.parameter = ''
        self.ax = None
        self.fig_result = None
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Policy(5), QtGui.QSizePolicy.Policy(5))
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        # try to analyse using auto center feature
        # self.try_auto_center()
        self.set_connections()
        self.has_image = False
        self.has_text = False
        #
        self.setWindowTitle(
            QtGui.QApplication.translate("Film2Dose", "StarShot Results", None, QtGui.QApplication.UnicodeUTF8))

    def set_connections(self):
        self.analyse_button.clicked.connect(self.on_analyse)
        self.auto_radio.clicked.connect(self.on_auto)
        self.auto_radio.setChecked(True)
        self.manual_radio.clicked.connect(self.on_manual)

    def on_auto(self):
        self.mode_select = 1

    def on_manual(self):
        self.mode_select = 0

    def on_analyse(self):
        try:
            if not self.has_text:
                self.show_dialog()
                self.has_text = True
            if not self.has_image:
                im, delta = self.image_widget.get_image()
                self.star_obj.set_image(im, delta)

            self.fig_result = self.star_obj.plot_report(parameter=self.parameter, flag=self.mode[self.mode_select],
                                                        widget=False)
            self.result_widget.set_figure(self.fig_result)
            self.result_widget.show_figure()
            self.result_widget.show()
        except:
            txt = "We did not find an even number of points in mode %s. Try a new center and radius in manual mode " % \
                  self.mode[self.mode_select]
            QtGui.QMessageBox.information(None, "Error!", txt)

    def try_auto_center(self):
        self.show_dialog()
        im, delta = self.image_widget.get_image()
        self.star_obj.set_image(im, delta)
        try:
            self.fig_result = self.star_obj.plot_report(parameter=self.parameter, flag=self.mode[self.mode_select])
            self.result_widget.set_figure(self.fig_result)
            self.result_widget.show_figure()
            self.result_widget.show()
        except:
            txt = "We did not find an even number of points in mode %s. Try a new center and radius in manual mode " % \
                  self.mode[self.mode_select]
            QtGui.QMessageBox.information(None, "Error!", txt)
            self.manual_radio.setChecked(True)
            self.mode_select = 0

    def show_dialog(self):
        text, ok = QtGui.QInputDialog.getText(self, 'Star shot',
                                              'Enter the name of Linac parameter (gantry, etc..)')

        if ok:
            self.parameter = text


class StarShotResultWidget(QtGui.QWidget):
    def __init__(self, *args, **kwargs):
        super(StarShotResultWidget, self).__init__(*args, **kwargs)

        self.fig = None
        self.canvas = None
        self.navigation_toolbar = None
        self.setWindowTitle(
            QtGui.QApplication.translate("Film2Dose", "StarShot Results", None, QtGui.QApplication.UnicodeUTF8))
        self.verticalLayout = QtGui.QVBoxLayout(self)
        self.setLayout(self.verticalLayout)

    def set_figure(self, fig):
        self.fig = fig

    def show_figure(self):
        self.canvas = FigureCanvas(self.fig)
        FigureCanvas.setSizePolicy(self, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.verticalLayout.addWidget(self.canvas)
        self.navigation_toolbar = Film2DoseToolbar(self.canvas, self)
        self.navigation_toolbar.setIconSize(QtCore.QSize(46, 46))
        self.verticalLayout.addWidget(self.navigation_toolbar)


class GammaHistogramWidget(QtGui.QWidget):
    def __init__(self, fig):
        super(GammaHistogramWidget, self).__init__(None)
        self.fig = fig
        self.canvas = None
        self.navigation_toolbar = None
        self.setWindowTitle(
            QtGui.QApplication.translate("Film2Dose", "Gamma Histogram", None, QtGui.QApplication.UnicodeUTF8))
        self.verticalLayout = QtGui.QVBoxLayout(self)
        self.setLayout(self.verticalLayout)
        self.show_figure()

    def set_figure(self, fig):
        self.fig = fig

    def show_figure(self):
        self.canvas = FigureCanvas(self.fig)
        FigureCanvas.setSizePolicy(self, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.verticalLayout.addWidget(self.canvas)
        self.navigation_toolbar = Film2DoseToolbar(self.canvas, self)
        self.navigation_toolbar.setIconSize(QtCore.QSize(46, 46))
        self.verticalLayout.addWidget(self.navigation_toolbar)


class GetPointCanvas(FigureCanvas, BlockingMouseInput):
    """Class to represent the FigureCanvas widget"""

    def __init__(self, fig, npoints):
        self.fig = fig
        self.cid = None
        self.ax = self.fig.add_subplot(111)
        self.npoints = 0
        self.clicks = []
        self.show_clicks = True
        self.marks = []
        self.cal_points = []
        self.npoints = npoints
        FigureCanvas.__init__(self, self.fig)
        # FigureCanvas.updateGeometry(self)

    def _on_click(self, event):
        """
             Matplotlib event Handler function.
        :param event: Matplotlib event
        """
        if event.xdata is not None and event.ydata is not None:
            self.add_event(event)
            if event.button == 1:
                self.mouse_event_add(event)
            elif event.button == 3:
                self.mouse_event_pop(event)

            print(len(self.clicks))
            if len(self.clicks) == self.npoints:
                self.fig.canvas.mpl_disconnect(self.cid)

    def get_points(self, timeout=10000, show_clicks=True, mouse_add=1,
                   mouse_pop=3, mouse_stop=2):

        blocking_mouse_input = Film2DoseBlockingMouseInput(self.fig,
                                                           mouse_add=mouse_add,
                                                           mouse_pop=mouse_pop,
                                                           mouse_stop=mouse_stop)
        return blocking_mouse_input(n=self.npoints, timeout=timeout,
                                    show_clicks=show_clicks)
