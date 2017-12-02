# GETTING dvh DATA FROM DOSE
from unittest import TestCase

from constraints.constraints import MayoConstraintConverter
from constraints.tests import pi
from core.types import ResultType, PriorityType


class TestMaxMeanDoseConstraint(TestCase):
    def test_constrain(self):
        # instantiate MayoConstraintConverter
        structure_name = 'PTV_70_3mm'
        converter = MayoConstraintConverter()
        constrain = 'Mean[Gy] <= 20'
        mdc = converter.convert_to_dvh_constraint(structure_name, PriorityType.IDEAL, constrain)
        constrain_result = mdc.constrain(pi)
        assert not constrain_result.is_success
        assert constrain_result.result_type == ResultType.ACTION_LEVEL_1

        constrain = 'Mean[Gy] <= 70'
        max_dc = converter.convert_to_dvh_constraint(structure_name, PriorityType.IDEAL, constrain)
        constrain_result = max_dc.constrain(pi)

        assert constrain_result.is_success
        assert constrain_result.result_type == ResultType.PASSED
