from core.types import PriorityType, ResultType, VolumePresentation
from tests.conftest import planning_item


def test_get_dose_compliment_at_volume(test_case, converter):
    constrain = 'DC95%[cGy]'
    # instantiate MayoConstraintConverter
    structure_name = 'PTV_70_3mm'
    constrain = 'DC95%[Gy] >= 60'
    mdc = converter.convert_to_dvh_constraint(structure_name, PriorityType.IDEAL, constrain)
    # assert correct volume units
    test_case.assertAlmostEqual(mdc.volume, 95 * VolumePresentation.relative)
    constrain_result = mdc.constrain(planning_item)
    assert constrain_result.result_type == ResultType.PASSED

    structure_name = 'PTV_70_3mm'
    constrain = 'DC95%[%] >= 87'
    mdc = converter.convert_to_dvh_constraint(structure_name, PriorityType.IDEAL, constrain)
    # assert correct volume units
    constrain_result = mdc.constrain(planning_item)
    assert constrain_result.result_type == ResultType.PASSED

    structure_name = 'PTV_70_3mm'
    constrain = 'DC655.0261147733513cc[Gy] >= 60'
    mdc = converter.convert_to_dvh_constraint(structure_name, PriorityType.IDEAL, constrain)
    # assert correct volume units
    test_case.assertAlmostEqual(mdc.volume, 655.0261147733513 * VolumePresentation.absolute_cm3)
    constrain_result = mdc.constrain(planning_item)
    assert constrain_result.result_type == ResultType.PASSED

