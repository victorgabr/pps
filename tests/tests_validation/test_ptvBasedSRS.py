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


def plot_dvh_comp(dvh_calc, dvh, title, dest_folder=''):
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
    if dest_folder:
        dest_file = os.path.join(dest_folder, title + '_PlanIQ')
        plt.savefig(dest_file + '.png', format='png', dpi=100)


if __name__ == '__main__':
    # Given
    # planIQ dvh
    dvh_ref_path = '/home/victor/Dropbox/Plan_Competition_Project/tests/tests_data/ptv_based_SRS/PTV Based - PlanIQ DVH - Eclipse VMAT - Body included - Ahmad March 24.txt'
    plan_iq_dvh = read_planiq_dvh(dvh_ref_path)

    # compare against SLICER DV
    plan_folder = r'/home/victor/Dropbox/Plan_Competition_Project/tests/tests_data/ptv_based_SRS'

    # dcm_files, flag = get_participant_folder_data(plan_folder)
    rtss = r'/home/victor/Dropbox/Plan_Competition_Project/tests/tests_data/ptv_based_SRS/RS.1.2.246.352.205.5644208623194161296.369175407993399980.dcm'
    rt_dose = '/home/victor/Dropbox/Plan_Competition_Project/tests/tests_data/ptv_based_SRS/RD.1.2.246.352.71.7.584747638204.1898898.20180324182516.dcm'
    ini_file = '/home/victor/Dropbox/Plan_Competition_Project/gui/cases/BrainMetSRSCase/PyPlanScoring.ini'
    setup_calculation_options = get_calculation_options(ini_file)

    structures = PyDicomParser(filename=rtss).GetStructures()
    rd_dcm = PyDicomParser(filename=rt_dose)
    dose = rd_dcm.get_dose_3d()
    plan_dict = {}

    # criteria file
    file_path = '/home/victor/Dropbox/Plan_Competition_Project/tests/tests_data/ptv_based_SRS/Scoring_criteria.xlsx'
    criteria = pd.read_excel(file_path, sheet_name='BrainSRS')

    # setup RT case
    rt_case_tmp = RTCase("SRS", 123, structures, criteria)
    dvh_calc = DVHCalculator(rt_case_tmp, setup_calculation_options)

    # when calculate DVH using pyplanScoring
    planning_obj = PyPlanningItem(plan_dict, rt_case_tmp, dose, dvh_calc)
    planning_obj.calculate_dvh()

    # # Calculating the score
    plan_eval = PlanEvaluation()
    plan_eval.criteria = criteria
    df = plan_eval.eval_plan(planning_obj)
    score = df['Raw score'].sum()

    # compare DVHs
    # TPS
    # dvhs = rd_dcm.GetDVHs()
    # # compare against Eclipse DVH
    # dvh_calculated = planning_obj.dvh_data
    # for roi_number in dvhs.keys():
    #     if roi_number in dvh_calculated:
    #         plot_dvh_comp1(dvh_calculated[roi_number], dvhs[roi_number], structures[roi_number]['name'], plan_folder)
    #
    # plt.show()

    # PlanIQ
    # compare against PLANIQ
    for roi_number, dvh in planning_obj.dvh_data.items():
        name = dvh['name']
        plot_dvh_comp(dvh, plan_iq_dvh, name, '')

    plt.show()

    dvh_data = planning_obj.dvh_data
    dvhs = [dvh_data[12], dvh_data[13], dvh_data[14], dvh_data[15], dvh_data[16]]
    dvhs = sorted(dvhs, key=lambda d: len(d['data']))
    dvh_total = dvh_data[17]
    # prealocate merged dvh array.
    max_dvh = dvhs[-1]
    merged_dvh = np.zeros(len(max_dvh['data']))

    # sum DVHS
    for dvh in dvhs:
        for i in range(len(dvh['data'])):
            current = dvh['data'][i]
            merged_dvh[i] += current

    x = np.arange(len(max_dvh['data'])) * float(max_dvh['scaling'])
    plt.plot(x, merged_dvh, label='merged')
    xx = np.arange(len(dvh_total['data'])) * float(dvh_total['scaling'])
    plt.plot(xx, dvh_total['data'], label='total')
    plt.plot(plan_iq_dvh.index, plan_iq_dvh['PTV_ALL'], label='planIQ')
    plt.legend()