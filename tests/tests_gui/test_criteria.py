# testing lung case
import os
import pandas as pd
import numpy as np

from constraints.metrics import RTCase, PyPlanningItem, PlanEvaluation
from core.calculation import DVHCalculator, PyStructure, DVHCalculation
from core.dicom_reader import PyDicomParser
from core.types import Dose3D
import quantities as pq
import matplotlib.pyplot as plt

from tests.conftest import DATA_DIR


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
    dest_file = os.path.join(dest_folder, title)
    # plt.savefig(dest_file, format='png', dpi=100)


def plot_dvh_comp1(dvh_calc, dvh, title, dest_folder):
    x_label = 'Dose [Gy]'
    y_label = 'Volume [cc]'
    plt.figure()
    x_calc = np.arange(len(dvh_calc['data'])) * float(dvh_calc['scaling'])
    x = np.arange(len(dvh['data'])) * float(dvh['scaling'])
    plt.plot(x_calc, dvh_calc['data'] , label='PyPlanScoring')
    plt.plot(x, dvh['data'], label='Eclipse')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.title(title)
    dest_file = os.path.join(dest_folder, title + '_Ecplipse')
    plt.savefig(dest_file, format='jpg', dpi=100)


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


def test_lung_case_dvh(setup_calculation_options):
    # Given plan DVH
    folder = os.path.join(DATA_DIR, 'lungSBRT')

    # DVH reference
    dvh_ref_path = os.path.join(folder, 'DVH TEXT 87.9.txt')
    plan_dvh = read_planiq_dvh(dvh_ref_path)

    # parse DICOM folder
    rp = os.path.join(folder, 'RP.1.2.246.352.71.5.584747638204.1034529.20180301221910.dcm')
    rs_dvh = os.path.join(folder, 'RS.1.2.246.352.205.5039724533480738438.3109367781599983491.dcm')
    rd = os.path.join(folder, 'RD.1.2.246.352.71.7.584747638204.1891868.20180301221910.dcm')

    plan_dict = PyDicomParser(filename=rp).GetPlan()
    structures = PyDicomParser(filename=rs_dvh).GetStructures()
    dose_values = PyDicomParser(filename=rd).get_dose_matrix()
    grid = PyDicomParser(filename=rd).get_grid_3d()
    dose = Dose3D(dose_values, grid, pq.Gy)


    # criteria file
    file_path = os.path.join(folder, 'Scoring_criteria_2018.xlsx')
    criteria = pd.read_excel(file_path, sheet_name='BiLateralLungSBRTCase')

    # setup RT case
    rt_case_tmp = RTCase("H&N", 123, structures, criteria)
    dvh_calc = DVHCalculator(rt_case_tmp, setup_calculation_options)

    # when calculate DVH using pyplanScoring
    planning_obj = PyPlanningItem(plan_dict, rt_case_tmp, dose, dvh_calc)
    planning_obj.calculate_dvh()

    # then Get dvh data and compare with plan_data
    # for roi_number, dvh in planning_obj.dvh_data.items():
    #     name = dvh['name']
    #     plot_dvh_comp(dvh, plan_dvh, name, folder)
    #
    # plt.show()



    #compare it abainst eclipse data
    # # compare with TPS DVH
    # dvhs =  PyDicomParser(filename=rd).GetDVHs()
    # # compare against Eclipse DVH
    # dvh_calculated = planning_obj.dvh_data
    # for roi_number in dvhs.keys():
    #     if roi_number in dvh_calculated:
    #         plot_dvh_comp1(dvh_calculated[roi_number], dvhs[roi_number], structures[roi_number]['name'], folder)
    #
    # plt.show()
    # lung SBRT case


    # Calculating the score
    plan_eval = PlanEvaluation()
    plan_eval.criteria = criteria
    df = plan_eval.eval_plan(planning_obj)

    assert df['Raw score'].sum() == 90.01



