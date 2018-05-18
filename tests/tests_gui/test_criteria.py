# testing lung case
import pandas as pd
import os
import numpy as np

from constraints.metrics import RTCase, PyPlanningItem, PlanEvaluation
from core.calculation import DVHCalculator, PyStructure, DVHCalculation
from core.dicom_reader import PyDicomParser
from core.io import get_participant_folder_data
from core.types import Dose3D
import quantities as pq
import matplotlib.pyplot as plt

from tests.conftest import DATA_DIR
from validation.validation import CurveCompare


def plot_dvh(dvh_calc, title):
    x_label = 'Dose [Gy]'
    y_label = 'Volume [cc]'
    plt.figure()
    x_calc = np.arange(len(dvh_calc['data'])) * float(dvh_calc['scaling'])

    plt.plot(x_calc, dvh_calc['data'], label='PyPlanScoring')
    plt.xlim([x_calc.min(), x_calc.max()])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.title(title)


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
    plt.savefig(dest_file + '.png', format='png', dpi=100)


def plot_dvh_comp1(dvh_calc, dvh, title, dest_folder):
    x_label = 'Dose [Gy]'
    y_label = 'Volume [cc]'
    plt.figure()
    x_calc = np.arange(len(dvh_calc['data'])) * float(dvh_calc['scaling'])
    x = np.arange(len(dvh['data'])) * float(dvh['scaling'])
    plt.plot(x_calc, dvh_calc['data'], label='PyPlanScoring')
    plt.plot(x, dvh['data'], label='Eclipse')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.title(title)
    dest_file = os.path.join(dest_folder, title + '_Ecplipse.png')
    plt.savefig(dest_file, format='png', dpi=100)


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


def test_dvh_comparisson(setup_calculation_options):
    # Given plan DVH
    folder = os.path.join(DATA_DIR, 'lungSBRT')

    # parse DICOM folder
    # plan_folder = r'C:\Users\Victor\Dropbox\Plan_Competition_Project\tests\tests_data\pinnacle\plan1'
    plan_folder = r'C:\Users\Victor\Dropbox\Plan_Competition_Project\tests\tests_data\pinnacle\plan2'
    dcm_files, flag = get_participant_folder_data(plan_folder)

    rs_dvh = os.path.join(folder, 'RS.dcm')

    plan_dict = PyDicomParser(filename=dcm_files['rtplan']).GetPlan()
    # structures = PyDicomParser(filename=dcm_files['rtss']).GetStructures()
    structures = PyDicomParser(filename=rs_dvh).GetStructures()
    rd_dcm = PyDicomParser(filename=dcm_files['rtdose'])
    dose = rd_dcm.get_dose_3d()

    # criteria file
    file_path = os.path.join(folder, 'Scoring_criteria.xlsx')
    criteria = pd.read_excel(file_path, sheet_name='BiLateralLungSBRTCase')

    # setup RT case
    rt_case_tmp = RTCase("SBRT", 123, structures, criteria)
    dvh_calc = DVHCalculator(rt_case_tmp, setup_calculation_options)

    # when calculate DVH using pyplanScoring
    planning_obj = PyPlanningItem(plan_dict, rt_case_tmp, dose, dvh_calc)
    planning_obj.calculate_dvh()

    # for k, v in planning_obj.dvh_data.items():
    #     plot_dvh(v, v['name'])
    #     plt.show()

    # # Calculating the score
    plan_eval = PlanEvaluation()
    plan_eval.criteria = criteria
    df = plan_eval.eval_plan(planning_obj)

    raw_score =  df['Raw score'].sum()
    pass

def test_lung_case_dvh(setup_calculation_options):
    # Given plan DVH
    folder = os.path.join(DATA_DIR, 'lungSBRT')

    # DVH reference
    dvh_ref_path = os.path.join(folder, 'PlanIQ DVH - Ahmad Plan - March 12.txt')
    plan_dvh = read_planiq_dvh(dvh_ref_path)

    # parse DICOM folder

    rp = os.path.join(folder, 'RP.dcm')
    rs_dvh = os.path.join(folder, 'RS.dcm')
    rd = os.path.join(folder, 'RD.dcm')

    plan_dict = PyDicomParser(filename=rp).GetPlan()
    structures = PyDicomParser(filename=rs_dvh).GetStructures()
    dose_values = PyDicomParser(filename=rd).get_dose_matrix()
    grid = PyDicomParser(filename=rd).get_grid_3d()
    dose = Dose3D(dose_values, grid, pq.Gy)

    # criteria file
    file_path = os.path.join(folder, 'Scoring_criteria.xlsx')
    criteria = pd.read_excel(file_path, sheet_name='BiLateralLungSBRTCase')

    # setup RT case
    rt_case_tmp = RTCase("SBRT", 123, structures, criteria)
    dvh_calc = DVHCalculator(rt_case_tmp, setup_calculation_options)

    # when calculate DVH using pyplanScoring
    planning_obj = PyPlanningItem(plan_dict, rt_case_tmp, dose, dvh_calc)
    planning_obj.calculate_dvh()

    # compare against PLANIQ
    for roi_number, dvh in planning_obj.dvh_data.items():
        name = dvh['name']
        plot_dvh_comp(dvh, plan_dvh, name, folder)

    # plt.show()

    # compare clinical DVH data
    #
    dvhs = PyDicomParser(filename=rd).GetDVHs()
    # compare against Eclipse DVH
    dvh_calculated = planning_obj.dvh_data
    res = {}
    res_eclipse = {}
    for roi_number in dvhs.keys():
        if roi_number in dvh_calculated:
            dvh_calc = dvh_calculated[roi_number]
            planiq_dvh = plan_dvh[structures[roi_number]['name']]

            x_calc = np.arange(len(dvh_calc['data'])) * float(dvh_calc['scaling'])
            x_planiq = plan_dvh.index
            cmp = CurveCompare(x_planiq, planiq_dvh, x_calc, dvh_calc['data'])
            # cmp.plot_results("PlanIQ", "PyPlanScoring", structures[roi_number]['name'])
            # plt.show()
            res[structures[roi_number]['name']] = cmp.stats_paper

            # Eclipse
            eclipse_dvh = dvhs[roi_number]
            x_eclipse = np.arange(len(eclipse_dvh['data'])) * float(eclipse_dvh['scaling'])

            cmp1 = CurveCompare(x_eclipse, eclipse_dvh['data'], x_calc, dvh_calc['data'])
            # cmp.plot_results("PlanIQ", "PyPlanScoring", structures[roi_number]['name'])
            res_eclipse[structures[roi_number]['name']] = cmp1.stats_paper

    res_plan_iq = pd.DataFrame.from_dict(res)
    res_plan_iq.to_csv(os.path.join(folder, "PyPlanScoringXPlanIQ.csv"))

    res_eclipsedf = pd.DataFrame.from_dict(res_eclipse)
    res_eclipsedf.to_csv(os.path.join(folder, "PyPlanScoringXEclipse.csv"))

    piq = [res_plan_iq.min().min(), res_plan_iq.max().max(), res_plan_iq.mean().mean()]
    ecl = [res_eclipsedf.min().min(), res_eclipsedf.max().max(), res_eclipsedf.mean().mean()]

    # # # then Get dvh data and compare with plan_data
    # for roi_number, dvh in planning_obj.dvh_data.items():
    #     name = dvh['name']
    #     plot_dvh_comp(dvh, plan_dvh, name, folder)
    #
    # plt.show()
    #
    # # compare it abainst eclipse data
    # # compare with TPS DVH
    dvhs = PyDicomParser(filename=rd).GetDVHs()
    # compare against Eclipse DVH
    dvh_calculated = planning_obj.dvh_data
    for roi_number in dvhs.keys():
        if roi_number in dvh_calculated:
            plot_dvh_comp1(dvh_calculated[roi_number], dvhs[roi_number], structures[roi_number]['name'], folder)

    # plt.show()
    # # lung SBRT case
    #
    # # Calculating the score
    plan_eval = PlanEvaluation()
    plan_eval.criteria = criteria
    df = plan_eval.eval_plan(planning_obj)

    assert df['Raw score'].sum() == 88.612
