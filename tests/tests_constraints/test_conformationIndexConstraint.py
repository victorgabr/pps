from pyplanscoring.core.types import PriorityType, ResultType
from tests.conftest import planning_item


def test_constrain(converter):
    structure_name = 'PTV70-BR.PLX 4MM'
    constrain = 'CI66.5Gy[] >= 0.6'

    min_ci = converter.convert_to_dvh_constraint(structure_name, PriorityType.IDEAL, constrain)
    constrain_result = min_ci.constrain(planning_item)
    assert str(min_ci) == constrain
    # assert not constrain_result.is_success
    # assert constrain_result.result_type == ResultType.ACTION_LEVEL_1
    assert constrain_result.is_success
    assert constrain_result.result_type == ResultType.PASSED
