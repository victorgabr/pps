import os
from unittest import TestCase

from pyplanscoring.core.constraints.metrics import PlanningItem, ConstrainMetric, MetricType
from pyplanscoring.core.dicomparser import ScoringDicomParser

DATA_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'test_data',
)

# DATA_DIR = r'C:\Users\vgalves\Dropbox\Plan_Competition_Project\pyplanscoring\core\constraints\tests\test_data'

filename = os.path.join(DATA_DIR, 'Scoring_criteria.xlsx')

rp = os.path.join(DATA_DIR, 'RP.dcm')
rs = os.path.join(DATA_DIR, 'RS.dcm')
rd = os.path.join(DATA_DIR, 'RD.dcm')

rp_dcm = ScoringDicomParser(filename=rp)
rs_dcm = ScoringDicomParser(filename=rs)
rd_dcm = ScoringDicomParser(filename=rd)


class TestConstrainMetric(TestCase):
    def test_metric_function(self):
        # test Min value constrain
        pi = PlanningItem(rp_dcm, rs_dcm, rd_dcm)
        structure_name = 'PTV70'
        query = 'D95%[Gy]'
        metric_type = MetricType.MIN
        target = [66.5, 64.0]
        max_score = 5
        cm = ConstrainMetric(structure_name, query, metric_type, target, max_score)
        res = cm.metric_function(pi)
        self.assertAlmostEqual(res, max_score)

        # test max value constrain
        structure_name = 'OPTIC CHIASM'
        query = 'Max[Gy]'
        metric_type = MetricType.MAX
        target = [52, 55]
        max_score = 4
        cm = ConstrainMetric(structure_name, query, metric_type, target, max_score)
        res = cm.metric_function(pi)
        self.assertAlmostEqual(res, max_score)
