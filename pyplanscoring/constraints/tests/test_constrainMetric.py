from unittest import TestCase

from constraints.metrics import ConstrainMetric, MetricType
from constraints.tests import pi


class TestConstrainMetric(TestCase):
    def test_metric_function(self):
        # test Min value constrain
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
