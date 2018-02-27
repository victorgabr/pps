import os
import pytest
from gui.tests import ini_file
# global variables
from constraints.metrics import RTCase
from core.calculation import DVHCalculator, get_calculation_options
from constraints.tests import DATA_DIR, rs_dcm, structures_dict
import pandas as pd

file_path = os.path.join(DATA_DIR, 'Scoring_criteria.xlsx')
criteria = pd.read_excel(file_path)
case_id = rs_dcm.GetSOPInstanceUID()
calculation_options = get_calculation_options(ini_file)
case = RTCase("H&N", case_id, structures_dict, criteria)

structure_names = ['PTV70 GH',
                   'PTV63 GH',
                   'PTV56-63 3MM',
                   'BODY',
                   'BRACHIAL PLEXUS',
                   'Brain',
                   'BRAINSTEM',
                   'BRAINSTEM PRV',
                   'CombinedParotids',
                   'CTV 56',
                   'CTV 63',
                   'CTV 70',
                   'ESOPHAGUS',
                   'GTV',
                   'LARYNX',
                   'LIPS',
                   'ANT CHAMBER LT',
                   'COCHLEA LT',
                   'EYE LT',
                   'LACRIMAL G. LT',
                   'LENS LT',
                   'OPTIC N. LT',
                   'PAROTID LT',
                   'MANDIBLE',
                   'OPTIC CHIASM',
                   'OPTIC CHIASM PRV',
                   'ORAL CAVITY',
                   'POST NECK',
                   'POST-CRICOID',
                   'PTV56',
                   'PTV63',
                   'PTV70',
                   'ANT CHAMBER RT',
                   'COCHLEA RT',
                   'EYE RT',
                   'LACRIMAL G. RT',
                   'LENS RT',
                   'OPTIC N. RT',
                   'PAROTID RT',
                   'SPINAL CORD',
                   'SPINAL CORD PRV',
                   'THYROID GLAND',
                   'OPTIC N. RT PRV',
                   'PTV63-70 3MM',
                   'PTV70-BR.PLX 4MM',
                   'OPTIC N. LT PRV',
                   'PTV63-BR.PLX 1MM']


@pytest.fixture(scope='module')
def rt_case():
    """

    :return: RTCase object
    """
    return case


@pytest.fixture(scope='module')
def dvh_calculator():
    return DVHCalculator(rt_case, calculation_options)
