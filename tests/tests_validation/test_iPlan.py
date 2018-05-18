import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from constraints.metrics import RTCase, PyPlanningItem, PlanEvaluation
from core.calculation import DVHCalculator, PyStructure, DVHCalculation
from core.calculation import get_calculation_options
from core.dicom_reader import PyDicomParser


def read_planiq_dvh(f):
    """
        Reads plan IQ exported txt DVH data
    :param f: path to file txt
    :return: Pandas Dataframe with DVH data in cGy and vol cc
    """
    with open(f, 'r') as io:
        txt = io.readlines()

    struc_header = [t.strip() for t in txt[0].split('\t') if len(t.strip()) > 0]
    data = np.asarray([t.split('\t') for t in txt[2:]], dtype=float)
    dose_axis = data[:, 0]
    idx = np.arange(1, data.shape[1], 2)
    vol = data[:, idx]

    plan_iq = pd.DataFrame(vol, columns=struc_header, index=dose_axis)

    return plan_iq


def plot_dvh_comp1(dvh_calc, dvh, title, dest_folder=''):
    x_label = 'Dose [Gy]'
    y_label = 'Volume [%]'
    plt.figure()
    x_calc = np.arange(len(dvh_calc['data'])) * float(dvh_calc['scaling'])
    x = np.arange(len(dvh['data'])) * float(dvh['scaling'])
    plt.plot(x_calc, dvh_calc['data'] / dvh_calc['data'][0] * 100, label='PyPlanScoring')
    plt.plot(x, dvh['data'] / dvh['data'][0] * 100, label='MultiPlan™')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.title(title)
    if dest_folder:
        dest_file = os.path.join(dest_folder, title + '_CK_Multiplan.png')
        plt.savefig(dest_file, format='png', dpi=100)


def plot_dvh(dvh_calc, title):
    x_label = 'Dose [Gy]'
    y_label = 'Volume [cc]'
    plt.figure()
    x_calc = np.arange(len(dvh_calc['data'])) * float(dvh_calc['scaling'])

    plt.plot(x_calc, dvh_calc['data'], label='PyPlanScoring')
    # plt.xlim([x_calc.min(), x_calc.max()])

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.title(title)


def read_slicer_dvh(file_path):
    df_slicer = pd.read_csv(file_path).dropna(axis=1)
    values_axis = df_slicer.iloc[:, 1::2]
    dose_axis = df_slicer.iloc[:, 0]
    columns = values_axis.columns
    volumes = np.array([re.findall('(\d+(?:\.\d+))', name)[0] for name in columns], dtype=float)
    values_axis = values_axis * volumes / 100
    values_axis['dose_axis'] = dose_axis
    slicer_dvh = values_axis.set_index('dose_axis')
    return slicer_dvh


def plot_dvh_comp(dvh_calc, dvh, title, dest_folder):
    x_label = 'Dose [Gy]'
    y_label = 'Volume [cc]'
    plt.figure()
    x_calc = np.arange(len(dvh_calc['data'])) * float(dvh_calc['scaling'])
    x = dvh.index
    plt.plot(x_calc, dvh_calc['data'], label='PyPlanScoring')
    plt.plot(x, dvh[title], label='PlanIQ')
    plt.xlim([x_calc.min(), x_calc.max()])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.title(title)
    dest_file = os.path.join(dest_folder, title + '_PlanIQ')
    plt.savefig(dest_file + '.png', format='png', dpi=100)


def read_iPlan_dvh(file_path_txt):
    df = pd.read_csv(iplan_dvh_file, sep='\t')
    data = df.iloc[1:, :].dropna(axis=1).astype(float)
    norm = df.iloc[0, :].dropna()
    norm = norm.apply(lambda x: float(x.strip('[%]')))
    dvh_iplan = data * norm / 100.0
    dvh_iplan = dvh_iplan.set_index('Dose')
    return dvh_iplan


if __name__ == '__main__':
    # Given
    # iPlan DVH
    iplan_dvh_file = '/home/victor/Dropbox/Plan_Competition_Project/tests/tests_data/brainlab_plan_ptv/dvh_ptv_based.txt'
    iplan_dvh = read_iPlan_dvh(iplan_dvh_file)

    # compare against SLICER DVH
    rtss = r'/home/victor/Dropbox/Plan_Competition_Project/gui/cases/BrainMetSRSCase/PTV/RS.dcm'
    rt_dose = '/home/victor/Dropbox/Plan_Competition_Project/tests/tests_data/brainlab_plan_ptv/RTDO0265.dcm'

    structures = PyDicomParser(filename=rtss).GetStructures()
    rd_dcm = PyDicomParser(filename=rt_dose)

    dose = rd_dcm.get_dose_3d()
    # structures_py = [PyStructure(s) for k, s in structures.items()]
    structures_py = [PyStructure(s) for k, s in structures.items()]
    # structures_py = [PyStructure(s, end_cap=s['thickness'] / 2) for k, s in structures.items()]
    grid = None
    up_grid = (0.1, 0.1, 0.1)
    grids = []
    for s in structures_py:
        print( len(np.unique(s.z_axis_delta)))
        if len(np.unique(s.z_axis_delta)) == 1  and s.volume < 20:
            grids.append(up_grid)
        else:
            grids.append(None)

    dvh_pyplan = {}
    for s, g in zip(structures_py, grids):
        dvh_calci = DVHCalculation(s, dose, calc_grid=g)
        dvh_l = dvh_calci.calculate(verbose=True)
        dvh_pyplan[dvh_l['name']] = dvh_l

    for k, dvh in dvh_pyplan.items():
        if dvh['name'] in iplan_dvh:
            fig, ax = plt.subplots()
            x_calc = np.arange(len(dvh['data'])) * float(dvh['scaling'])
            vols = dvh['data']
            ax.plot(x_calc, vols, label='PyPlanScoring')
            iplan_dvh[dvh['name']].plot(ax=ax)
            plt.legend()
            plt.show()

    plt.close('all')