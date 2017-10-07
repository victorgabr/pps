import os
from unittest import TestCase

import numpy.testing as npt
import pandas as pd

from pyplanscoring.core.constraints.metrics import PlanEvaluation, PlanningItem
from pyplanscoring.core.dicomparser import ScoringDicomParser
from pyplanscoring.core.dvhcalculation import load

DATA_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'test_data',
)

# DATA_DIR = r'C:\Users\vgalves\Dropbox\Plan_Competition_Project\pyplanscoring\core\constraints\tests\test_data'

filename = os.path.join(DATA_DIR, 'Scoring_criteria.xlsx')

rp = os.path.join(DATA_DIR, 'RP.dcm')
rs = os.path.join(DATA_DIR, 'RS.dcm')
rd = os.path.join(DATA_DIR, 'RD.dcm')
dvh_path = os.path.join(DATA_DIR, 'PyPlanScoring_dvh.dvh')

rp_dcm = ScoringDicomParser(filename=rp)
rs_dcm = ScoringDicomParser(filename=rs)
rd_dcm = ScoringDicomParser(filename=rd)

# report data pyplanscoring
plan_report = os.path.join(DATA_DIR, 'plan_report.xlsx')

report_df = pd.read_excel(plan_report, skiprows=31)
pyplan_dvh = load(dvh_path)
dvh = pyplan_dvh['DVH']


class TestPlanEvaluation(TestCase):
    def test_eval_plan(self):
        # test using eclipse DVH data
        filename = os.path.join(DATA_DIR, 'Scoring_criteria.xlsx')
        plan_eval = PlanEvaluation()
        plan_eval.read(filename)

        pi = PlanningItem(rp_dcm, rs_dcm, rd_dcm)
        df = plan_eval.eval_plan(pi)
        self.assertAlmostEqual(df['Raw score'].sum(), 76.097797709986182)

        # using pyplanscoring data
        # Ci calculated from DVH data
        pi1 = PlanningItem(rp_dcm, rs_dcm, rd_dcm)
        pi1.dvh_data = dvh
        df1 = plan_eval.eval_plan(pi1)
        self.assertAlmostEqual(df1['Raw score'].sum(), 74.288403194174222, places=1)

    def test_failed_structures(self):
        pi = PlanningItem(rp_dcm, rs_dcm, rd_dcm)
        struc_name = 'LIPS'
        mayo_format_query = 'D0.1cc[Gy]'
        dose = pi.execute_query(mayo_format_query, struc_name)
        mayo_format_query = 'D0.1%[Gy]'
        dose1 = pi.execute_query(mayo_format_query, struc_name)

        # set pyplanscoring DVH
        pi.dvh_data = dvh

        mayo_format_query = 'D0.1cc[Gy]'
        dose3 = pi.execute_query(mayo_format_query, struc_name)
        mayo_format_query = 'D0.1%[Gy]'
        dose4 = pi.execute_query(mayo_format_query, struc_name)
        a = 1
        # found bug at old criteria sheet.

        # test again using PyPlanScoring DVH data
        filename = os.path.join(DATA_DIR, 'Scoring_criteria_old.xlsx')
        plan_eval = PlanEvaluation()
        plan_eval.read(filename)

        # using pyplanscoring data
        # Ci calculated from DVH data
        pi1 = PlanningItem(rp_dcm, rs_dcm, rd_dcm)
        pi1.dvh_data = dvh
        df1 = plan_eval.eval_plan(pi1)
        total_score = df1['Raw score'].sum()
        self.assertAlmostEqual(total_score, 73.9531905344, places=1)

        ref_data = report_df['Result'].ix[:30]
        calc_data = df1['Result']
        diff = calc_data - ref_data
        diff.index = df1['Structure Name']
        # TODO CHECK DIFFERENCES
        # Check difference in PAROTID LD DOSE
        npt.assert_array_almost_equal(calc_data, ref_data, decimal=1)
