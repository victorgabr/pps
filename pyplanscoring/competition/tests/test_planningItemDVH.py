from unittest import TestCase

from competition.tests import high_score
# oseValuePresentation
# from constraints import QueryExtensions
from constraints.query import QueryExtensions
from core.types import DoseValuePresentation
from pyplanscoring.competition.statistical_dvh import PlanningItemDVH

plan_dvh = PlanningItemDVH(plan_dvh=high_score)


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

    def test_query_ci_stats(self):
        pi = PlanningItemDVH(plan_dvh=high_score)
        # confomity index calculation
        query_str = 'CI66.5Gy[]'
        target_name = 'PTV70-BR.PLX 4MM'
        mc = QueryExtensions()
        mc.read(query_str)
        ci = pi.query_ci_stats(mc, target_name)
        # fantasy Friedemann Herberth  - FANTASY - 21 APRIL FINAL - 100.0
        self.assertAlmostEqual(ci, 0.924595466114167, places=1)
