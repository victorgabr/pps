import os
from unittest import TestCase

import pandas as pd
import pytest
import quantities as pq

from pyplanscoring.constraints.constraints import MayoConstraintConverter
from pyplanscoring.constraints.metrics import (PlanningItem, PyPlanningItem,
                                               RTCase)
from pyplanscoring.constraints.query import QueryExtensions
from pyplanscoring.core.calculation import (DVHCalculator,
                                            get_calculation_options)
from pyplanscoring.core.dicom_reader import PyDicomParser
from pyplanscoring.core.types import Dose3D, DVHData

# TODO review fixtures changing to singleton objects
# TODO split this file in many conftest.py inside each folder
# TODO REFACTOR ALL TESTS TO LUNG_SBRT CASE WITH CONTOURS WITH HOLES only

DATA_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'tests_data',
)

ini_file = os.path.join(DATA_DIR, "PyPlanScoring.ini")
file_path = os.path.join(DATA_DIR, 'Scoring_criteria.xlsx')
criteria = pd.read_excel(file_path)
calculation_options = get_calculation_options(ini_file)


@pytest.fixture()
def ini_file_path():
    return os.path.join(DATA_DIR, "PyPlanScoring.ini")


@pytest.fixture()
def setup_calculation_options():
    return get_calculation_options(ini_file)


@pytest.fixture()
def data_dir():
    return DATA_DIR


@pytest.fixture()
def dicom_folder():
    return os.path.join(DATA_DIR, 'lungSBRT')


# criteria 2018
@pytest.fixture()
def criteria_lung():
    file_path = os.path.join(DATA_DIR, 'Scoring_criteria_2018.xlsx')
    return pd.read_excel(file_path, sheet_name='BiLateralLungSBRTCase')


rs = os.path.join(DATA_DIR, 'RS.dcm')
rd = os.path.join(DATA_DIR, 'RD.dcm')
rp = os.path.join(DATA_DIR, 'RP.dcm')

str_names = [
    'LENS LT', 'PAROTID LT', 'BRACHIAL PLEXUS', 'OPTIC N. RT PRV',
    'OPTIC CHIASM PRV', 'OPTIC N. RT', 'ORAL CAVITY', 'BRAINSTEM',
    'SPINAL CORD', 'OPTIC CHIASM', 'LENS RT', 'LARYNX', 'SPINAL CORD PRV',
    'EYE LT', 'PTV56', 'BRAINSTEM PRV', 'PTV70', 'OPTIC N. LT PRV', 'EYE RT',
    'PTV63', 'OPTIC N. LT', 'LIPS', 'ESOPHAGUS', 'PTV70'
]

rs_dvh = os.path.join(DATA_DIR, 'RS_dvh.dcm')
structures_tmp = PyDicomParser(filename=rs_dvh).GetStructures()
to_index = {v['name']: k for k, v in structures_tmp.items()}


# fixtures
@pytest.fixture(scope="session")
def rp_dcm():
    """
        Fixture to return DICOM-RTPLAN obj
    :return:
    """
    return PyDicomParser(filename=rp)


@pytest.fixture(scope="session")
def rs_dcm():
    """
        Fixture to return DICOM-RTSTRUCT obj
    :return:
    """
    return PyDicomParser(filename=rs)


@pytest.fixture(scope="session")
def rd_dcm():
    """
        Fixture to return DICOM-RTDOSE obj
    :return:
    """
    return PyDicomParser(filename=rd)


@pytest.fixture(scope="session")
def dose_3d():
    dose_values = PyDicomParser(filename=rd).get_dose_matrix()
    grid = PyDicomParser(filename=rd).get_grid_3d()

    return Dose3D(dose_values, grid, pq.Gy)


@pytest.fixture(scope="session")
def structures():
    """
        Return structure contours from dicom
    :rtype: dict
    :return:
    """
    return structures_tmp


@pytest.fixture(scope="session")
def py_planning_item():
    plan_dict = PyDicomParser(filename=rp).GetPlan()
    rs_dvh = os.path.join(DATA_DIR, 'RS_dvh.dcm')
    structures_tmp1 = PyDicomParser(filename=rs_dvh).GetStructures()
    criteria1 = pd.read_excel(file_path, sheet_name='calc_dvh')
    rt_case_tmp = RTCase("H&N", 123, structures_tmp1, criteria1)
    dose_values = PyDicomParser(filename=rd).get_dose_matrix()
    grid = PyDicomParser(filename=rd).get_grid_3d()
    dose = Dose3D(dose_values, grid, pq.Gy)
    dvh_calc = DVHCalculator(rt_case_tmp, calculation_options)

    return PyPlanningItem(plan_dict, rt_case_tmp, dose, dvh_calc)


@pytest.fixture(scope="session")
def brain():
    """
        Return brain structure dict obj
    :return:
    """
    return structures_tmp[6]


@pytest.fixture(scope="session")
def ptv70():
    return structures_tmp[to_index['PTV70']]


@pytest.fixture(scope="session")
def lens():
    return structures_tmp[to_index['LENS LT']]


@pytest.fixture(scope="session")
def spinal_cord():
    return structures_tmp[to_index['SPINAL CORD']]


@pytest.fixture(scope="session")
def parotid_lt():
    return structures_tmp[to_index['PAROTID LT']]


@pytest.fixture(scope="session")
def optic_chiasm():
    return structures_tmp[to_index['OPTIC CHIASM']]


@pytest.fixture(scope="session")
def body():
    return structures_tmp[4]


@pytest.fixture(scope="session")
def plot_flag():
    # plot flag
    return False
    # return True


@pytest.fixture(scope="session")
def rt_case():
    """
        Return RT case object
    :return:
    """
    return RTCase("H&N", 123, structures_tmp, criteria)


@pytest.fixture(scope="session")
def dvh_calculator():
    rs_dvh = os.path.join(DATA_DIR, 'RS_dvh.dcm')
    structures_tmp1 = PyDicomParser(filename=rs_dvh).GetStructures()
    rt_case_tmp = RTCase("H&N", 123, structures_tmp1, criteria)
    return DVHCalculator(rt_case_tmp, calculation_options)


@pytest.fixture()
def dvh():
    rd_dvh = os.path.join(DATA_DIR, 'RD_dvh.dcm')
    dvh_all = PyDicomParser(filename=rd_dvh).GetDVHs()
    return DVHData(dvh_all[61])


@pytest.fixture(scope="session")
def dvh1():
    rd_dvh = os.path.join(DATA_DIR, 'RD_dvh.dcm')
    dvh_all = PyDicomParser(filename=rd_dvh).GetDVHs()
    return DVHData(dvh_all[61])


@pytest.fixture(scope="session")
def query_extensions():
    return QueryExtensions()


#
# @pytest.fixture(scope="module")
# def planning_item():
#     rd_dvh = os.path.join(DATA_DIR, 'RD_dvh.dcm')
#     rs_dvh = os.path.join(DATA_DIR, 'RS_dvh.dcm')
#     pi = PlanningItem(PyDicomParser(filename=rp),
#                       PyDicomParser(filename=rs_dvh),
#                       PyDicomParser(filename=rd_dvh))
#     return pi

rd_dvh = os.path.join(DATA_DIR, 'RD_dvh.dcm')
rs_dvh = os.path.join(DATA_DIR, 'RS_dvh.dcm')
planning_item = PlanningItem(
    PyDicomParser(filename=rp),
    PyDicomParser(filename=rs_dvh),
    PyDicomParser(filename=rd_dvh))


@pytest.fixture(scope="session")
def converter():
    return MayoConstraintConverter()


@pytest.fixture(scope="session")
def test_case():
    return TestCase()


test_oc = structures_tmp[to_index['OPTIC CHIASM']]
