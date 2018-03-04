from unittest import TestCase

from constraints.constraints import MayoConstraintConverter
from constraints.tests import pi
from core.types import PriorityType, VolumePresentation, ResultType

converter = MayoConstraintConverter()


class TestVolumeAtDoseConstraint(TestCase):
    def test_volume(self):
        structure_name = 'PTV70'
        constrain = 'V6103.854532025905cGy[%] >= 95'
        mdc = converter.convert_to_dvh_constraint(structure_name, PriorityType.IDEAL, constrain)
        # assert correct volume units
        self.assertAlmostEqual(mdc.volume, 95 * VolumePresentation.relative)
        constrain_result = mdc.constrain(pi)
        assert constrain_result.is_success
        assert constrain_result.result_type == ResultType.PASSED

        constrain = 'V6103.854532025905cGy[%] >= 97'
        mdc = converter.convert_to_dvh_constraint(structure_name, PriorityType.IDEAL, constrain)
        # assert correct volume units
        self.assertAlmostEqual(mdc.volume, 97 * VolumePresentation.relative)
        constrain_result = mdc.constrain(pi)
        assert constrain_result.is_success
        # assert constrain_result.result_type == ResultType.ACTION_LEVEL_1

        constrain = 'V6103.854532025905cGy[%] <= 94'
        mdc = converter.convert_to_dvh_constraint(structure_name, PriorityType.IDEAL, constrain)
        # assert correct volume units
        self.assertAlmostEqual(mdc.volume, 94 * VolumePresentation.relative)
        constrain_result = mdc.constrain(pi)
        assert not constrain_result.is_success
        assert constrain_result.result_type == ResultType.ACTION_LEVEL_1

        constrain = 'V6103.854532025905cGy[cc] <= 655.0261147733513'
        mdc = converter.convert_to_dvh_constraint(structure_name, PriorityType.IDEAL, constrain)
        # assert correct volume units
        self.assertAlmostEqual(mdc.volume, 655.0261147733513 * VolumePresentation.absolute_cm3)
        constrain_result = mdc.constrain(pi)

        assert constrain_result.is_success
        assert constrain_result.result_type == ResultType.PASSED
