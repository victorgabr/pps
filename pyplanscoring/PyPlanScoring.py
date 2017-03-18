from __future__ import division

import configparser
import os
import platform
import sys

import matplotlib
from PySide import QtGui, QtCore

# TODO comment this lines before compile using pyinstaller
matplotlib.use('Qt4Agg')
matplotlib.rcParams['backend.qt4'] = 'PySide'

from pyplanscoring import PyPlanScoringQT
from pyplanscoring.core.dosimetric import read_scoring_criteria
from pyplanscoring.core.scoring import Participant, get_participant_folder_data

__version__ = '0.0.1'
__author__ = 'Dr. Victor Gabriel Leandro Alves, D.Sc.'
__copyright__ = "Copyright (C) 2004 Victor Gabriel Leandro Alves"
__license__ = "Licenced for evaluation purposes only"


def _sys_getenc_wrapper():
    return 'UTF-8'


sys.getfilesystemencoding = _sys_getenc_wrapper

# set globals
# SET COMPETITION 2017

folder = os.getcwd()
path = os.path.join(folder, 'Scoring Criteria.txt')
constrains, scores, criteria = read_scoring_criteria(path)
banner_path = os.path.join(folder, '2017 Plan Comp Banner.jpg')

# Get calculation defaults
config = configparser.ConfigParser()
config.read(os.path.join(folder, 'PyPlanScoring.ini'))
calculation_options = dict()
calculation_options['end_cap'] = config.getfloat('DEFAULT', 'end_cap')
calculation_options['use_tps_dvh'] = config.getboolean('DEFAULT', 'use_tps_dvh')
calculation_options['up_sampling'] = config.getboolean('DEFAULT', 'up_sampling')
calculation_options['maximum_upsampled_volume_cc'] = config.getfloat('DEFAULT', 'maximum_upsampled_volume_cc')
calculation_options['voxel_size'] = config.getfloat('DEFAULT', 'voxel_size')
calculation_options['num_cores'] = config.getint('DEFAULT', 'num_cores')
calculation_options['save_dvh_figure'] = config.getboolean('DEFAULT', 'save_dvh_figure')
calculation_options['save_dvh_data'] = config.getboolean('DEFAULT', 'save_dvh_data')
calculation_options['mp_backend'] = config['DEFAULT']['mp_backend']


class MainDialog(QtGui.QMainWindow, PyPlanScoringQT.Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainDialog, self).__init__(parent=parent)
        self.setupUi(self)
        self.participant = None
        self.folder_root = None
        self.files_data = None
        self.name = ''
        self.set_conections()
        self.result = None

    def set_conections(self):
        self.action_developer.triggered.connect(self.about)
        self.import_button.clicked.connect(self.on_import)
        self.save_reports_button.clicked.connect(self.on_save)

    @QtCore.Slot(object)
    def worker_done(self, obj):
        out_name = '_plan_scoring_report.xls'
        self.participant, self.result = obj
        self.listWidget.addItem(str('Plan Score: %1.3f' % self.result))
        out_file = os.path.join(self.folder_root, self.name + out_name)
        self.participant.save_score(out_file, banner_path=banner_path)
        self.listWidget.addItem(str('Saving report on %s ' % out_file))

    def on_import(self):
        self.listWidget.clear()
        self.name = self.lineEdit.text()
        if self.name:
            self.folder_root = QtGui.QFileDialog.getExistingDirectory(self,
                                                                      "Select the directory containing only: RP and RD Dicom RT dose files from one plan",
                                                                      QtCore.QDir.currentPath())

            if self.folder_root:
                truth, self.files_data = get_participant_folder_data(self.name, self.folder_root)
                if truth:
                    self.listWidget.addItem(str('Loaded %s - Plan Files:' % self.name))
                    self.listWidget.addItems(self.files_data.index.astype(str))
                else:
                    msg = "<p>missing Dicom Files: " + self.files_data.to_string()
                    QtGui.QMessageBox.critical(self, "Missing Data", msg, QtGui.QMessageBox.Abort)
        else:
            msg = "Please set participant's name"
            QtGui.QMessageBox.critical(self, "Missing Data", msg, QtGui.QMessageBox.Abort)

    def _calc_score(self):
        rd = self.files_data.reset_index().set_index(1).ix['rtdose']['index']
        rp = self.files_data.reset_index().set_index(1).ix['rtplan']['index']
        rs = os.path.join(folder, 'RS.1.2.246.352.71.4.584747638204.253443.20170222200317.dcm')

        if calculation_options['use_tps_dvh']:
            rs = self.files_data.reset_index().set_index(1).ix['rtss']['index']

        self.participant = Participant(rp, rs, rd, calculation_options=calculation_options)
        self.participant.set_participant_data(self.name)
        val = self.participant.eval_score(constrains_dict=constrains, scores_dict=scores, criteria_df=criteria)

        return val

    def on_save(self):
        self.listWidget.addItem(str('-------------Calculating score--------------'))
        if calculation_options['use_tps_dvh']:
            self.listWidget.addItem(str('Using TPS exported DVH'))
            self.listWidget.addItem(str('Matched RS/RD dicom files'))
            self.listWidget.addItem(str(self.files_data.reset_index().set_index(1).ix['rtss']['index']))
            self.listWidget.addItem(str(self.files_data.reset_index().set_index(1).ix['rtdose']['index']))

        out_name = '_plan_scoring_report.xlsx'
        sc = self._calc_score()
        self.listWidget.addItem(str('Plan Score: %1.3f' % sc))
        out_file = os.path.join(self.folder_root, self.name + out_name)
        self.participant.save_score(out_file, banner_path=banner_path, report_header=self.name)
        self.listWidget.addItem(str('Saving report on %s ' % out_file))

    def about(self):
        txt = "PlanReport - H&N Nasopharynx - 2017 RT Plan Competition: %s \n" \
              "Be the strongest link in the radiotherapy chain\n" \
              "https://radiationknowledge.org \n" \
              "Author: %s\n" \
              "Copyright (C) 2017 Victor Gabriel Leandro Alves, All rights reserved\n" \
              "Platform details: Python %s on %s\n" \
              "This program aims to calculate an approximate score only.\n" \
              "your final score may be different due to structure boundaries and dose interpolation uncertainties\n" \
              "%s" \
              % (__version__, __author__, platform.python_version(), platform.system(), __license__)

        QtGui.QMessageBox.about(self, 'Information', txt)


def main():
    app = QtGui.QApplication(sys.argv)
    form = MainDialog()
    form.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    form = MainDialog()
    form.show()
    sys.exit(app.exec_())
