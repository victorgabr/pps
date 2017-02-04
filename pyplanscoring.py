import os
import platform
import sys

import PySide
from PySide import QtGui, QtCore

import PyPlanScoringQT
from dosimetric import read_scoring_criteria
from scoring import Participant, get_participant_folder_data

__version__ = '0.0.1'


def _sys_getenc_wrapper():
    return 'UTF-8'


sys.getfilesystemencoding = _sys_getenc_wrapper

# SET COMPETITION 2017
folder = os.getcwd()
f_2017 = 'PlanIQ Criteria TPS PlanIQ matched str names - TXT Fromat - Last mod Jan23.txt'
path = os.path.join(folder, f_2017)
constrains, scores, criteria = read_scoring_criteria(path)


class MainDialog(QtGui.QMainWindow, PyPlanScoringQT.Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainDialog, self).__init__(parent=parent)
        self.setupUi(self)
        self.participant = None
        self.folder_root = None
        self.files_data = None
        self.name = ''
        self.set_conections()

    def set_conections(self):
        self.action_developer.triggered.connect(self.about)
        self.import_button.clicked.connect(self.on_import)
        self.save_reports_button.clicked.connect(self.on_save)

    def on_import(self):
        self.listWidget.clear()
        self.name = self.lineEdit.text()
        if self.name:
            self.folder_root = QtGui.QFileDialog.getExistingDirectory(self,
                                                                      "Select the directory containing only: RP, RS and RD Dicom RT dose files from one plan",
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
        rs = self.files_data.reset_index().set_index(1).ix['rtss']['index']
        # end cap and upsample only small structures
        self.participant = Participant(rp, rs, rd, upsample='_up_sampled_', end_cap=True)
        self.participant.set_participant_data(self.name)
        val = self.participant.eval_score(constrains_dict=constrains, scores_dict=scores, criteria_df=criteria)
        return val

    def on_save(self):
        self.listWidget.addItem(str('-------------Calculating score--------------'))
        sc = self._calc_score()
        self.listWidget.addItem(str('Plan Score: %1.3f' % sc))
        out_file = os.path.join(self.folder_root, self.name + '_plan_scoring_report.xls')
        self.participant.save_score(out_file)
        self.listWidget.addItem(str('Saving report on %s ' % out_file))

    def about(self):
        txt = "<b> PyPlanScoring - H&N Cancer: %s </b>" \
              "<p> https://radiationknowledge.org/" \
              "<p>Developer: Dr. Victor Gabriel Leandro Alves, D.Sc." \
              "<p> Copyright &copy; 2017 Victor Gabriel Leandro Alves, " \
              "All rights reserved" \
              " <p>Platform details:<p> Python %s - PySide version %s - Qt version %s on %s" % (
                  __version__, platform.python_version(), PySide.__version__, PySide.QtCore.__version__,
                  platform.system())

        QtGui.QMessageBox.about(self, 'Information', txt)


app = QtGui.QApplication(sys.argv)
form = MainDialog()
form.show()
sys.exit(app.exec_())
