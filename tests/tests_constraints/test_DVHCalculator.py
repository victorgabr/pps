"""
Test cases for DVHCalculator class

"""


# import os
#
# from constraints.metrics import RTCase
# from core.calculation import DVHCalculator
# from core.dicom_reader import PyDicomParser
# from tests.conftest import DATA_DIR, criteria, calculation_options
#
# rs_dvh = os.path.join(DATA_DIR, 'RS_dvh.dcm')
# structures_tmp1 = PyDicomParser(filename=rs_dvh).GetStructures()
# rt_case_tmp = RTCase("H&N", 123, structures_tmp1, criteria)
# d_calc = DVHCalculator(rt_case_tmp, calculation_options)
#

def test_init(dvh_calculator):
    assert dvh_calculator.calculation_options


def test_voxel_size(dvh_calculator):
    assert isinstance(dvh_calculator.voxel_size, tuple)
    assert 0.5 in dvh_calculator.voxel_size


def test_calc_setup(dvh_calculator):
    # should be a list
    strucs, grids = dvh_calculator.calculation_setup
    assert len(strucs) == len(grids)


def test_calculate_mp(dvh_calculator, dose_3d):
    dvh_data = dvh_calculator.calculate_mp(dose_3d)
    assert dvh_data
