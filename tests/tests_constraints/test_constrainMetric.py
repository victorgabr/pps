from constraints.metrics import ConstrainMetric, MetricType
from tests.conftest import planning_item


def test_metric_function(test_case):
    # test Min value constrain
    structure_name = 'PTV70'
    query = 'D95%[Gy]'
    metric_type = MetricType.MIN
    target = [66.5, 64.0]
    max_score = 5
    cm = ConstrainMetric(structure_name, query, metric_type, target, max_score)
    res = cm.metric_function(planning_item)
    test_case.assertAlmostEqual(res, max_score)

    # test max value constrain
    structure_name = 'OPTIC CHIASM'
    query = 'Max[Gy]'
    metric_type = MetricType.MAX
    target = [52, 55]
    max_score = 4
    cm = ConstrainMetric(structure_name, query, metric_type, target, max_score)
    res = cm.metric_function(planning_item)
    test_case.assertAlmostEqual(res, max_score)
