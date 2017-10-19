import os
from unittest import TestCase

from pyplanscoring.competition.statistical_dvh import PlanningItemDVH
from pyplanscoring.core.constraints.types import DoseValuePresentation
from pyplanscoring.core.dvhcalculation import load

DATA_DIR = r'C:\Users\Victor\Dropbox\Plan_Competition_Project\pyplanscoring\core\constraints\tests\test_data'
rp = os.path.join(DATA_DIR, 'RP.dcm')
rs = os.path.join(DATA_DIR, 'RS.dcm')
rd = os.path.join(DATA_DIR, 'RD.dcm')

dvh_path = os.path.join(DATA_DIR, 'PyPlanScoring_dvh.dvh')

pyplan_dvh = load(dvh_path)
dvh = pyplan_dvh['DVH']

plan_dvh = PlanningItemDVH(plan_dvh=dvh)


class TestPlanningItemDVH(TestCase):
    def test_dose_value_presentation(self):
        assert plan_dvh.dose_value_presentation == DoseValuePresentation.Absolute

    def test_contains_structure(self):
        s = 'SPINAL CORD'
        t, _ = plan_dvh.contains_structure(s)
        assert t
        s1 = 'spinal cord'
        t1, _ = plan_dvh.contains_structure(s1)
        assert t1

        s2 = 'ANIFANF'
        t2, _ = plan_dvh.contains_structure(s2)
        assert not t2

    def test_get_structure(self):
        s = 'SPINAL CORD'
        res = plan_dvh.get_structure(s)
        assert res
        s1 = 'spinal cord'
        t1 = plan_dvh.get_structure(s1)
        assert t1

        s2 = 'ANIFANF'
        t2 = plan_dvh.get_structure(s2)
        assert "Structure ANIFANF not found"

    def test_get_dvh_cumulative_data(self):
        s = 'SPINAL CORD'
        res = plan_dvh.get_dvh_cumulative_data(s)
        assert res
        s1 = 'spinal cord'
        t1 = plan_dvh.get_dvh_cumulative_data(s1)
        assert t1

        s2 = 'ANIFANF'
        t2 = plan_dvh.get_dvh_cumulative_data(s2)
        assert "Structure ANIFANF not found"
