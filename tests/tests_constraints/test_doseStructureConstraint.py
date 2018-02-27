from core.types import PriorityType
from tests.conftest import planning_item


def test_can_constrain(converter):
    structure_name = 'PTV70-BR.PLX 4MM'
    constrain = 'CI66.5Gy[] >= 0.6'

    max_dc = converter.convert_to_dvh_constraint(structure_name, PriorityType.IDEAL, constrain)
    can_constraint_result = max_dc.can_constrain(planning_item)
    assert can_constraint_result.is_success
