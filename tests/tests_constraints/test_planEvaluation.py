import os

import numpy.testing as npt
import pandas as pd

from constraints.metrics import PlanEvaluation
from core.dvhcalculation import load
from tests.conftest import DATA_DIR, planning_item

filename = os.path.join(DATA_DIR, 'Scoring_criteria.xlsx')
dvh_path = os.path.join(DATA_DIR, 'PyPlanScoring_dvh.dvh')
# report data pyplanscoring
plan_report = os.path.join(DATA_DIR, 'plan_report.xlsx')
report_df = pd.read_excel(plan_report, skiprows=31)
pyplan_dvh = load(dvh_path)
dvh = pyplan_dvh['DVH']


def test_eval_plan(test_case):
    # test using eclipse DVH data
    filename = os.path.join(DATA_DIR, 'Scoring_criteria.xlsx')
    plan_eval = PlanEvaluation()
    crit = plan_eval.read(filename)

    df = plan_eval.eval_plan(planning_item)
    test_case.assertAlmostEqual(df['Raw score'].sum(), 76.097797709986182)

    # using pyplanscoring data
    # Ci calculated from DVH data
    pi1 = planning_item
    pi1.dvh_data = dvh
    df1 = plan_eval.eval_plan(pi1)
    test_case.assertAlmostEqual(df1['Raw score'].sum(), 74.288403194174222, places=1)


def test_failed_structures(test_case):
    struc_name = 'LIPS'
    mayo_format_query = 'D0.1cc[Gy]'
    dose = planning_item.execute_query(mayo_format_query, struc_name)
    mayo_format_query = 'D0.1%[Gy]'
    dose1 = planning_item.execute_query(mayo_format_query, struc_name)

    # set pyplanscoring DVH
    planning_item.dvh_data = dvh

    mayo_format_query = 'D0.1cc[Gy]'
    dose3 = planning_item.execute_query(mayo_format_query, struc_name)
    mayo_format_query = 'D0.1%[Gy]'
    dose4 = planning_item.execute_query(mayo_format_query, struc_name)
    a = 1
    # found bug at old criteria sheet.

    # test again using PyPlanScoring DVH data
    filename = os.path.join(DATA_DIR, 'Scoring_criteria_old.xlsx')
    plan_eval = PlanEvaluation()
    plan_eval.read(filename)

    # using pyplanscoring data
    # Ci calculated from DVH data
    pi1 = planning_item
    pi1.dvh_data = dvh
    df1 = plan_eval.eval_plan(pi1)
    total_score = df1['Raw score'].sum()
    test_case.assertAlmostEqual(total_score, 73.9531905344, places=1)

    ref_data = report_df['Result'].ix[:30]
    calc_data = df1['Result']
    diff = calc_data - ref_data
    diff.index = df1['Structure Name']
    # TODO CHECK DIFFERENCES
    # Check difference in PAROTID LD DOSE
    npt.assert_array_almost_equal(calc_data, ref_data, decimal=1)
