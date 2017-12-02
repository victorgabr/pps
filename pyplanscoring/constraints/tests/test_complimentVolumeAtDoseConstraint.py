from unittest import TestCase

from constraints.constraints import MayoConstraintConverter
from constraints.tests import pi
from core.types import PriorityType, ResultType

converter = MayoConstraintConverter()


class TestComplimentVolumeAtDoseConstraint(TestCase):
    def test_volume(self):
        structure_name = 'PTV_70_3mm'
        constrain = 'CV6103.854532025905cGy[%] >= 4'
        mdc = converter.convert_to_dvh_constraint(structure_name, PriorityType.IDEAL, constrain)
        constrain_result = mdc.constrain(pi)
        assert constrain_result.is_success
        assert constrain_result.result_type == ResultType.PASSED

        constrain = 'CV6103.854532025905cGy[%] >= 7'
        mdc = converter.convert_to_dvh_constraint(structure_name, PriorityType.IDEAL, constrain)
        constrain_result = mdc.constrain(pi)
        assert not constrain_result.is_success
        assert constrain_result.result_type == ResultType.ACTION_LEVEL_1

        structure_name = 'PTV_70_3mm'
        constrain = 'CV6103.854532025905cGy[%] <= 4'
        mdc = converter.convert_to_dvh_constraint(structure_name, PriorityType.IDEAL, constrain)
        constrain_result = mdc.constrain(pi)
        assert not constrain_result.is_success
        assert constrain_result.result_type == ResultType.ACTION_LEVEL_1
