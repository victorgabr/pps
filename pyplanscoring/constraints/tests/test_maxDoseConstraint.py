# GETTING dvh DATA FROM DOSE
from unittest import TestCase

from constraints.constraints import MayoConstraintConverter
from constraints.tests import pi
from core.types import ResultType, PriorityType


class TestMaxDoseConstraint(TestCase):
    def test_constrain(self):
        # instantiate MayoConstraintConverter
        converter = MayoConstraintConverter()
        constrain = 'Max[Gy] <= 45'
        structure_name = 'SPINAL CORD'
        mdc = converter.convert_to_dvh_constraint(structure_name, PriorityType.IDEAL, constrain)
        constrain_result = mdc.constrain(pi)
        assert constrain_result.is_success
        assert constrain_result.result_type == ResultType.PASSED

        constrain = 'Max[Gy] <= 20'
        max_dc = converter.convert_to_dvh_constraint(structure_name, PriorityType.IDEAL, constrain)
        constrain_result = max_dc.constrain(pi)
        assert not constrain_result.is_success
        assert constrain_result.result_type == ResultType.ACTION_LEVEL_1
