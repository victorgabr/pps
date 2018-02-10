import os
from unittest import TestCase

import pandas as pd

from constraints.metrics import RTCase
from constraints.tests import DATA_DIR, rs_dcm, structures_dict

# global variables
file_path = os.path.join(DATA_DIR, 'Scoring_criteria.xlsx')
criteria = pd.read_excel(file_path)

case_id = rs_dcm.GetSOPInstanceUID()

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


class TestRTCase(TestCase):
    def test_metrics(self):
        assert not case.metrics.empty
        # self.fail()

    def test_structures(self):
        assert case.structures

    def test_name(self):
        assert case.name

    def test_case_id(self):
        assert case.case_id

    def test_get_structure(self):
        struc = case.get_structure('PTV56')
        assert struc

        struc1 = case.get_structure('Spinal Cord')
        assert struc1

        struc2 = case.get_structure('spinal cord')
        assert struc2

        struc3 = case.get_structure('SPINAL CORD')
        assert struc3

        struc4 = case.get_structure('SPINAL Coord')
        assert struc4

        struc = case.get_structure('XSUGUA')
        assert struc == "Structure XSUGUA not found"
