# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
import os
import sys
import subprocess
import site

ui = ['Film2doseMainWindow.ui', 'evo_widget.ui', 'PicketFence.ui', 'starshot.ui', 'tps_widget.ui', 'DoseComp.ui',
      'dose_conversion.ui', 'dose_optim.ui', 'edit_grid.ui', 'fit_curves_dialog.ui', 'fit_curves_widget.ui',
      'fit_mode_widget.ui', 'FormImageUI.ui', 'get_cal_points.ui']

qt = ['MainWindowQt.py', 'OptimizationQT.py', 'PicketFenceQT.py', 'StarShotQT.py', 'TPSWidgetQT.py',
      'DoseCompQT.py', 'DoseConversionQT.py', 'DoseOptimizedQT.py', 'EditGridQT.py', 'FitCurvesDialogQT.py',
      'FitCurvesQt.py', 'FitModeQT.py', 'FormImageQT.py', 'GetCalPointsQT.py']


def create_qrc(folder, name):
    """
        Create a QT qrc file collecting all files in a folder
    :param folder: Path to resources
    :param name: output filename (*.qrc)

    format:
        <!DOCTYPE RCC><RCC version="1.0">
        <qresource>
            <file>icons/arch_linux.png</file>
            <file>icons/App_icon.png</file>
            <file>icons/export.png</file>
            <file>icons/fedora.png</file>
            <file>icons/scanner.png</file>
            <file>icons/windows.png</file>
             <file>icons/app.png</file>
        </qresource>
        </RCC>
    """
    # TODO debug scrip for python 3
    files = os.listdir(folder)
    print(files)
    with open(name, 'w') as f:
        f.write('<!DOCTYPE RCC><RCC version="1.0">\n')
        f.write('<qresource>\n')
        for icon in files:
            tag = '<file>' + folder[1:] + '/%s</file>\n' % icon
            f.write(tag)
        f.write('</qresource>\n')
        f.write('</RCC>\n')


def create_python_qrc(qrfile):
    """
        Converts a *.qrc file into a python QT resource file using pyside-rcc.exe
    :param qrfile: Qt resource XML file  *.qrc
    """
    if sys.platform != 'linux':
        path = site.getsitepackages()[1] + r'\PySide\pyside-rcc.exe'
        f = qrfile.split('.')[0] + '_rc.py'
        print(path)
        cmd = [path, '-py3', '-o', f, qrfile]
        subprocess.call(cmd)
    else:
        path = 'pyside-rcc'
        path = r'/home/victor/miniconda3/lib/python3.4/site-packages/PySide/pyside-rcc'
        f = qrfile.split('.')[0] + '_rc.py'
        cmd = [path, '-py3', '-o', f, qrfile]
        subprocess.call(cmd)


def qt2py(uifile, outfile):
    """
        Generates from QT-UI files from QT-Designer to Python files
    :param uifile: Qt-Designer file, *.ui
    :param outfile: Python file output file *.py
    """
    if sys.platform != 'linux':
        # path = site.getsitepackages()[1] + r'\PySide\scripts\uic.py'
        # ex = 'python ' + path + ' ' + '-o' + ' ' + outfile + ' ' + uifile
        # print(ex)
        # os.system(ex)
        ex = r'C:\Miniconda3\Scripts\pyside-uic.exe'
        cmd = [ex, '-o', outfile, uifile]
        print(cmd)
        subprocess.call(cmd)

    else:
        cmd = ['/home/victor/miniconda3/bin/pyside-uic', '-o', outfile, uifile]
        subprocess.call(cmd)


def generate_all():
    ui = ['Film2doseMainWindow.ui', 'evo_widget.ui', 'PicketFence.ui', 'starshot.ui', 'tps_widget.ui', 'DoseComp.ui',
          'dose_conversion.ui', 'dose_optim.ui', 'edit_grid.ui', 'fit_curves_dialog.ui', 'fit_curves_widget.ui',
          'fit_mode_widget.ui', 'FormImageUI.ui', 'get_cal_points.ui']

    qt = ['MainWindowQt.py', 'OptimizationQT.py', 'PicketFenceQT.py', 'StarShotQT.py', 'TPSWidgetQT.py',
          'DoseCompQT.py', 'DoseConversionQT.py', 'DoseOptimizedQT.py', 'EditGridQT.py', 'FitCurvesDialogQT.py',
          'FitCurvesQt.py', 'FitModeQT.py', 'FormImageQT.py', 'GetCalPointsQT.py']

    conversion = dict(zip(ui, qt))

    for k in conversion:
        out_ui = conversion[k]
        ui_file = os.path.join(os.getcwd(), k)
        print(ui_file)
        out_file = os.path.join(os.getcwd(), out_ui)
        print(out_file)

        qt2py(ui_file, out_file)


if __name__ == "__main__":


    uifile = r'C:\Users\vgalves\Dropbox\Plan_Competition_Project\PyPlanScoring.ui'
    outfile = r'C:\Users\vgalves\Dropbox\Plan_Competition_Project\PyPlanScoringQT.py'
    qt2py(uifile, outfile)





    # save_average_image()

    # print(os.getcwd())

    # qrfile_name = 'icons.qrc'
    # qrfile = os.path.join(os.getcwd(), qrfile_name)
    # print(qrfile)
    # icons_folder = os.path.join(os.getcwd(), 'icons')
    #
    # print(icons_folder)
    # print(os.getcwd())
    # create_python_qrc(qrfile)

    # # generate_all()
    #
    # qt_ui = 'PicketFence.ui'
    # out_ui = 'PicketFenceQT.py'
    # ui_file = os.path.join(os.getcwd(), qt_ui)
    # print(ui_file)
    # out_file = os.path.join(os.getcwd(), out_ui)
    # print(out_file)
    # #
    # qt2py(ui_file, out_file)


# save average images
#
