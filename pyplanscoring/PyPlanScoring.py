from __future__ import division

import os
import platform
import sys

import matplotlib
from PySide import QtGui, QtCore

# from PySide.QtCore import QLocale

matplotlib.use('Qt4Agg')
matplotlib.rcParams['backend.qt4'] = 'PySide'

from pyplanscoring import PyPlanScoringQT
from pyplanscoring.dosimetric import read_scoring_criteria
from pyplanscoring.scoring import Participant, get_participant_folder_data

__version__ = '0.0.1'
__author__ = 'Dr. Victor Gabriel Leandro Alves, D.Sc.'
__copyright__ = "Copyright (C) 2004 Victor Gabriel Leandro Alves"
__license__ = "Licenced for evaluation purposes only"


def _sys_getenc_wrapper():
    return 'UTF-8'


sys.getfilesystemencoding = _sys_getenc_wrapper

# SET COMPETITION 2017
folder = os.getcwd()
path = os.path.join(folder, 'Scoring Criteria.txt')
constrains, scores, criteria = read_scoring_criteria(path)
banner_path = os.path.join(folder, '2017 Plan Comp Banner.jpg')
rs = os.path.join(folder, 'RS.1.2.246.352.71.4.584747638204.253443.20170222200317.dcm')


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

        # rs = self.files_data.reset_index().set_index(1).ix['rtss']['index']
        # end cap and upsample only small structures
        self.participant = Participant(rp, rs, rd, upsample='_up_sampled_', end_cap=True)
        self.participant.set_participant_data(self.name)
        arg = QtCore.QEventLoop.AllEvents
        QtCore.QCoreApplication.processEvents(arg, maxtime=1000)
        val = self.participant.eval_score(constrains_dict=constrains, scores_dict=scores, criteria_df=criteria,
                                          dicom_dvh=False)

        return val

    def on_save(self):
        self.listWidget.addItem(str('-------------Calculating score--------------'))
        # self.worker.set_parameters(self.name,
        #                            self.files_data,
        #                            constrains,
        #                            scores,
        #                            criteria,
        #                            dicom_dvh=False,
        #                            upsample=True,
        #                            end_cap=True)
        #
        # self.worker.show()
        # self.worker.run()
        # # self.worker_thread.start()

        out_name = '_plan_scoring_report.xlsx'
        # if self.tps_check_box.isChecked():
        #     self.listWidget.addItem(str('Using TPS calculated DVH from DICOM-RT dose file'))
        #     out_name = '_plan_scoring_report_TPS_DVH.xlsx'

        sc = self._calc_score()
        self.listWidget.addItem(str('Plan Score: %1.3f' % sc))
        out_file = os.path.join(self.folder_root, self.name + out_name)
        self.participant.save_score(out_file, banner_path=banner_path)
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


class Worker(QtGui.QWidget):
    calc_done = QtCore.Signal(object)

    def __init__(self, parent=None):
        super(Worker, self).__init__(parent)
        self.participant = None
        self.constrains = None
        self.scores = None
        self.criteria = None
        self.dicom_dvh = None
        self.result = None

    def set_parameters(self, name, files_data, constrains, scores, criteria, dicom_dvh, upsample, end_cap):
        rd = files_data.reset_index().set_index(1).ix['rtdose']['index']
        rp = files_data.reset_index().set_index(1).ix['rtplan']['index']
        rs = files_data.reset_index().set_index(1).ix['rtss']['index']
        # end cap and upsample only small structures
        up = ''
        if upsample:
            up = '_up_sampled_'

        self.participant = Participant(rp, rs, rd, upsample=up, end_cap=end_cap)
        self.participant.set_participant_data(name)
        self.constrains = constrains
        self.scores = scores
        self.criteria = criteria
        self.dicom_dvh = dicom_dvh

    @QtCore.Slot()
    def run(self):
        self.result = self.participant.eval_score(constrains_dict=self.constrains,
                                                  scores_dict=self.scores,
                                                  criteria_df=self.criteria,
                                                  dicom_dvh=self.dicom_dvh)

        self.calc_done.emit((self.participant, self.result))

    def get_result(self):
        return self.participant, self.result


def main():
    app = QtGui.QApplication(sys.argv)
    form = MainDialog()
    form.show()
    QtCore.QCoreApplication.processEvents()
    # sys.exit(app.exec_())
    app.exec_()


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    form = MainDialog()
    form.show()
    # QtCore.QCoreApplication.processEvents()
    sys.exit(app.exec_())
    # app.exec_()
