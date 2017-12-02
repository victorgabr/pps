from unittest import TestCase

from constraints.constraints import MayoConstraintConverter
from constraints.tests import pi
from core.types import PriorityType


class TestDoseStructureConstraint(TestCase):
    def test_can_constrain(self):
        structure_name = 'PTV70-BR.PLX 4MM'
        constrain = 'CI66.5Gy[] >= 0.6'
        converter = MayoConstraintConverter()
        max_dc = converter.convert_to_dvh_constraint(structure_name, PriorityType.IDEAL, constrain)
        can_constraint_result = max_dc.can_constrain(pi)
        assert can_constraint_result.is_success
