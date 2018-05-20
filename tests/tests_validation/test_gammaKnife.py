import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pyplanscoring.core.calculation import PyStructure, DVHCalculation
from pyplanscoring.core.dicom_reader import PyDicomParser
from pyplanscoring.core.io import get_participant_folder_data


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


def test_gk_dvh():
    # Given
    slicer_dvh_file = r'C:\Users\Victor\Dropbox\Plan_Competition_Project\tests\tests_data\gk_plan\SLICER_RT_DVH.csv'
    plan_folder = r'C:\Users\Victor\Dropbox\Plan_Competition_Project\tests\tests_data\gk_plan'

    slicer_dvh = read_slicer_dvh(slicer_dvh_file)

    # strip columns names
    slicer_dvh.columns = ['Skull Value', 'CerebelarDir', 'FrontalEsq ', 'Tumor', 'Tumor 2',
                          'Tumor 3', 'Tumor 4', 'Tumor 5']

    dcm_files, flag = get_participant_folder_data(plan_folder)
    plan_dict = PyDicomParser(filename=dcm_files['rtplan']).GetPlan()
    structures = PyDicomParser(filename=dcm_files['rtss']).GetStructures()
    rd_dcm = PyDicomParser(filename=dcm_files['rtdose'])
    dose = rd_dcm.get_dose_3d()
    structures_py = [PyStructure(s, end_cap=s['thickness'] / 2) for k, s in structures.items()]

    grid = (0.1, 0.1, 0.1)
    dvh_pyplan = {}
    for s in structures_py[:-1]:
        dvh_calci = DVHCalculation(s, dose, calc_grid=grid)
        dvh_l = dvh_calci.calculate(verbose=True)
        dvh_pyplan[dvh_l['name']] = dvh_l
        # plot_dvh(dvh_l, dvh_l['name'])

    for target_name, dvh_calc in dvh_pyplan.items():
        dose_t5 = slicer_dvh.index
        t5 = slicer_dvh.loc[:, target_name].values
        x_calc = np.arange(len(dvh_calc['data'])) * float(dvh_calc['scaling'])
        py_t5 = dvh_calc['data']

        plt.plot(dose_t5, t5, label='Slicer-RT')
        plt.plot(x_calc, py_t5, label='PyPlanScoring')
        plt.title(target_name)
        plt.legend()
        plt.show()
