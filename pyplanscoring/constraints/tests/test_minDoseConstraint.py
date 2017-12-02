# GETTING dvh DATA FROM DOSE
from unittest import TestCase

from constraints.constraints import MayoConstraintConverter
from constraints.tests import pi
from core.types import ResultType, PriorityType


class TestMinDoseConstraint(TestCase):
    def test_constrain(self):
        # instantiate MayoConstraintConverter
        converter = MayoConstraintConverter()
        constrain = 'Min[Gy] >= 20'
        structure_name = 'PTV_70_3mm'
        mdc = converter.convert_to_dvh_constraint(structure_name, PriorityType.IDEAL, constrain)
        constrain_result = mdc.constrain(pi)
        assert constrain_result.is_success
        assert constrain_result.result_type == ResultType.PASSED

        constrain = 'Min[Gy] >= 70'
        max_dc = converter.convert_to_dvh_constraint(structure_name, PriorityType.IDEAL, constrain)
        constrain_result = max_dc.constrain(pi)
        assert not constrain_result.is_success
        assert constrain_result.result_type == ResultType.ACTION_LEVEL_1
