from core.types import PriorityType, ResultType
from tests.conftest import planning_item


def test_constrain(converter):
    # instantiate MayoConstraintConverter

    constrain = 'Max[Gy] <= 45'
    structure_name = 'SPINAL CORD'
    mdc = converter.convert_to_dvh_constraint(structure_name, PriorityType.IDEAL, constrain)
    constrain_result = mdc.constrain(planning_item)
    assert constrain_result.is_success
    assert constrain_result.result_type == ResultType.PASSED

    constrain = 'Max[Gy] <= 20'
    max_dc = converter.convert_to_dvh_constraint(structure_name, PriorityType.IDEAL, constrain)
    constrain_result = max_dc.constrain(planning_item)
    assert not constrain_result.is_success
    assert constrain_result.result_type == ResultType.ACTION_LEVEL_1
