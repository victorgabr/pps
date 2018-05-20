import os

from pyplanscoring.core.calculation import get_calculation_options


def test_get_calculation_options(data_dir):
    ini_file = os.path.join(data_dir, "PyPlanScoring.ini")

    calc_options = get_calculation_options(ini_file)
    assert calc_options
