from unittest import TestCase

from constraints.constraints import MayoConstraintConverter
from constraints.tests import pi
from core.types import PriorityType, ResultType


class TestHomogeneityIndexConstraint(TestCase):
    def test_constrain(self):
        structure_name = 'PTV70-BR.PLX 4MM'
        constrain = 'HI70Gy[] <= 0.08'
        converter = MayoConstraintConverter()
        max_dc = converter.convert_to_dvh_constraint(structure_name, PriorityType.IDEAL, constrain)
        constrain_result = max_dc.constrain(pi)
        assert not constrain_result.is_success
        assert constrain_result.result_type == ResultType.ACTION_LEVEL_1
