from unittest import TestCase

from constraints.constraints import MayoConstraintConverter
from constraints.tests import pi
from core.types import PriorityType, ResultType

converter = MayoConstraintConverter()


class TestConformationIndexConstraint(TestCase):
    def test_constrain(self):
        structure_name = 'PTV70-BR.PLX 4MM'
        constrain = 'CI66.5Gy[] >= 0.6'
        converter = MayoConstraintConverter()
        min_ci = converter.convert_to_dvh_constraint(structure_name, PriorityType.IDEAL, constrain)
        constrain_result = min_ci.constrain(pi)
        assert str(min_ci) == constrain
        # assert not constrain_result.is_success
        # assert constrain_result.result_type == ResultType.ACTION_LEVEL_1
        assert constrain_result.is_success
        assert constrain_result.result_type == ResultType.PASSED
