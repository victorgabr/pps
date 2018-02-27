"""
Test cases for DVHCalculator class

"""


def test_init(dvh_calculator):
    assert dvh_calculator.calculation_options


def test_voxel_size(dvh_calculator):
    assert isinstance(dvh_calculator.voxel_size, tuple)
    assert 0.2 in dvh_calculator.voxel_size


def test_get_grid_array(dvh_calculator):
    # should be a list
    arr = dvh_calculator.get_grid_array()
    # assert isinstance(arr, list)
