from unittest import TestCase

from constraints.constraints import MayoConstraintConverter
from constraints.tests import pi
from core.types import PriorityType, ResultType, VolumePresentation

converter = MayoConstraintConverter()


class TestComplimentDoseAtVolumeConstraint(TestCase):
    def test_get_dose_compliment_at_volume(self):
        'DC95%[cGy]'
        # instantiate MayoConstraintConverter
        structure_name = 'PTV_70_3mm'
        constrain = 'DC95%[Gy] >= 60'
        mdc = converter.convert_to_dvh_constraint(structure_name, PriorityType.IDEAL, constrain)
        # assert correct volume units
        self.assertAlmostEqual(mdc.volume, 95 * VolumePresentation.relative)
        constrain_result = mdc.constrain(pi)
        assert constrain_result.result_type == ResultType.PASSED

        structure_name = 'PTV_70_3mm'
        constrain = 'DC95%[%] >= 87'
        mdc = converter.convert_to_dvh_constraint(structure_name, PriorityType.IDEAL, constrain)
        # assert correct volume units
        constrain_result = mdc.constrain(pi)
        assert constrain_result.result_type == ResultType.PASSED

        structure_name = 'PTV_70_3mm'
        constrain = 'DC655.0261147733513cc[Gy] >= 60'
        mdc = converter.convert_to_dvh_constraint(structure_name, PriorityType.IDEAL, constrain)
        # assert correct volume units
        self.assertAlmostEqual(mdc.volume, 655.0261147733513 * VolumePresentation.absolute_cm3)
        constrain_result = mdc.constrain(pi)
        assert constrain_result.result_type == ResultType.PASSED
