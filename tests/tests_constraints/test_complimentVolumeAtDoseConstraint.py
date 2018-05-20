from pyplanscoring.core.types import PriorityType, ResultType
from tests.conftest import planning_item


def test_volume(converter):
    structure_name = 'PTV_70_3mm'
    constrain = 'CV6103.854532025905cGy[%] >= 4'
    mdc = converter.convert_to_dvh_constraint(structure_name, PriorityType.IDEAL, constrain)
    constrain_result = mdc.constrain(planning_item)
    assert constrain_result.is_success
    assert constrain_result.result_type == ResultType.PASSED

    constrain = 'CV6103.854532025905cGy[%] >= 7'
    mdc = converter.convert_to_dvh_constraint(structure_name, PriorityType.IDEAL, constrain)
    constrain_result = mdc.constrain(planning_item)
    assert not constrain_result.is_success
    assert constrain_result.result_type == ResultType.ACTION_LEVEL_1

    structure_name = 'PTV_70_3mm'
    constrain = 'CV6103.854532025905cGy[%] <= 4'
    mdc = converter.convert_to_dvh_constraint(structure_name, PriorityType.IDEAL, constrain)
    constrain_result = mdc.constrain(planning_item)
    assert not constrain_result.is_success
    assert constrain_result.result_type == ResultType.ACTION_LEVEL_1
