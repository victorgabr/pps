from unittest import TestCase

from constraints.constraints import MayoConstraintConverter
from constraints.tests import pi
from core.types import PriorityType, ResultType, VolumePresentation

converter = MayoConstraintConverter()


class TestDoseAtVolumeConstraint(TestCase):
    def test_volume(self):
        # instantiate MayoConstraintConverter
        structure_name = 'PTV70'
        constrain = 'D95%[Gy] >= 66'
        mdc = converter.convert_to_dvh_constraint(structure_name, PriorityType.IDEAL, constrain)
        # assert correct volume units
        self.assertAlmostEqual(mdc.volume, 95 * VolumePresentation.relative)
        constrain_result = mdc.constrain(pi)
        assert constrain_result.result_type == ResultType.PASSED

        constrain = 'D451.46456771249984cc[Gy] >= 65 '
        mdc = converter.convert_to_dvh_constraint(structure_name, PriorityType.IDEAL, constrain)
        # assert correct volume units
        self.assertAlmostEqual(mdc.volume, 451.46456771249984 * VolumePresentation.absolute_cm3)
        constrain_result = mdc.constrain(pi)
        assert constrain_result.result_type == ResultType.PASSED

    def test_volume_type(self):
        # instantiate MayoConstraintConverter
        structure_name = 'PTV70'
        constrain = 'D95%[Gy] >= 60'
        mdc = converter.convert_to_dvh_constraint(structure_name, PriorityType.IDEAL, constrain)
        # assert correct volume units
        self.assertAlmostEqual(mdc.volume_type, VolumePresentation.relative)

        constrain = 'D655.0261147733513cc[Gy] >= 65 '
        mdc = converter.convert_to_dvh_constraint(structure_name, PriorityType.IDEAL, constrain)
        # assert correct volume units
        self.assertAlmostEqual(mdc.volume_type, VolumePresentation.absolute_cm3)

    def test_passing_func(self):
        # instantiate MayoConstraintConverter
        structure_name = 'PTV70'
        constrain = 'D95%[Gy] >= 60'
        mdc = converter.convert_to_dvh_constraint(structure_name, PriorityType.IDEAL, constrain)
        constrain_result = mdc.constrain(pi)
        assert constrain_result.result_type == ResultType.PASSED

        constrain = 'D451.46456771249984cc[Gy] >= 70 '
        mdc = converter.convert_to_dvh_constraint(structure_name, PriorityType.IDEAL, constrain)
        constrain_result = mdc.constrain(pi)
        assert constrain_result.result_type == ResultType.ACTION_LEVEL_1

        # Not Passing Results
        constrain = 'D0.01cc[Gy] <= 45'
        structure_name = 'SPINAL CORD'
        mdc = converter.convert_to_dvh_constraint(structure_name, PriorityType.IDEAL, constrain)
        constrain_result = mdc.constrain(pi)
        assert constrain_result.result_type == ResultType.PASSED

        constrain = 'D0.01cc[Gy] <= 20'
        structure_name = 'SPINAL CORD'
        mdc = converter.convert_to_dvh_constraint(structure_name, PriorityType.IDEAL, constrain)
        constrain_result = mdc.constrain(pi)
        assert constrain_result.result_type == ResultType.ACTION_LEVEL_1
