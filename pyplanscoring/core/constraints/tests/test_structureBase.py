import os
from unittest import TestCase

import numpy.testing as npt

from pyplanscoring.core.constraints.types import StructureBase
from pyplanscoring.core.dicomparser import ScoringDicomParser

DATA_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'test_data',
)

DATA_DIR = r'C:\Users\Victor\Dropbox\Plan_Competition_Project\pyplanscoring\core\constraints\tests\test_data'

rs = os.path.join(DATA_DIR, 'RS.dcm')
rs_dcm = ScoringDicomParser(filename=rs)

structure_set = rs_dcm.GetStructures()

# BrainStem

structure_dict = structure_set[7]
structure_obj = StructureBase(structure_dict)


class TestStructureBase(TestCase):
    def test_center_point(self):
        npt.assert_array_almost_equal(structure_obj.center_point, [-3.42, -214.3, 61.5])

    def test_color(self):
        npt.assert_array_almost_equal(structure_obj.color, [0., 255., 255.])

    def test_dicom_type(self):
        assert structure_obj.dicom_type == 'ORGAN'

    def test_is_high_resolution(self):
        assert not structure_obj.is_high_resolution

    def test_roi_number(self):
        assert structure_obj.roi_number == 7

    def test_id(self):
        assert structure_obj.id == 'BRAINSTEM'

    def test_get_contours_on_image_plane(self):
        assert structure_obj.get_contours_on_image_plane('19.50')
        assert not structure_obj.get_contours_on_image_plane('19.5')
