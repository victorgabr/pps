from core.types import VolumePresentation, PriorityType, ResultType
from tests.conftest import planning_item


def test_volume(test_case, converter):
    structure_name = 'PTV70'
    constrain = 'V6103.854532025905cGy[%] >= 95'
    mdc = converter.convert_to_dvh_constraint(structure_name, PriorityType.IDEAL, constrain)
    # assert correct volume units
    test_case.assertAlmostEqual(mdc.volume, 95 * VolumePresentation.relative)
    constrain_result = mdc.constrain(planning_item)
    assert constrain_result.is_success
    assert constrain_result.result_type == ResultType.PASSED

    constrain = 'V6103.854532025905cGy[%] >= 97'
    mdc = converter.convert_to_dvh_constraint(structure_name, PriorityType.IDEAL, constrain)
    # assert correct volume units
    test_case.assertAlmostEqual(mdc.volume, 97 * VolumePresentation.relative)
    constrain_result = mdc.constrain(planning_item)
    assert constrain_result.is_success
    # assert constrain_result.result_type == ResultType.ACTION_LEVEL_1

    constrain = 'V6103.854532025905cGy[%] <= 94'
    mdc = converter.convert_to_dvh_constraint(structure_name, PriorityType.IDEAL, constrain)
    # assert correct volume units
    test_case.assertAlmostEqual(mdc.volume, 94 * VolumePresentation.relative)
    constrain_result = mdc.constrain(planning_item)
    assert not constrain_result.is_success
    assert constrain_result.result_type == ResultType.ACTION_LEVEL_1

    constrain = 'V6103.854532025905cGy[cc] <= 655.0261147733513'
    mdc = converter.convert_to_dvh_constraint(structure_name, PriorityType.IDEAL, constrain)
    # assert correct volume units
    test_case.assertAlmostEqual(mdc.volume, 655.0261147733513 * VolumePresentation.absolute_cm3)
    constrain_result = mdc.constrain(planning_item)

    assert constrain_result.is_success
    assert constrain_result.result_type == ResultType.PASSED
