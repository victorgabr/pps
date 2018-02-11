import os
from unittest import TestCase

from core.calculation import PyStructure, DVHCalculationMP
from core.io import IOHandler
from core.tests import ptv70, brain, body, lens, spinal_cord, dose_3d, DATA_DIR

# calculating a DVH
grid_up = (0.2, 0.2, 0.2)
structures_dicom = [lens, body, brain, ptv70, spinal_cord]
structures_py = [PyStructure(s) for s in structures_dicom]
grids = [grid_up, None, None, None, None]

calc_mp = DVHCalculationMP(dose_3d, structures_py, grids, verbose=True)
dvh_data = calc_mp.calculate_dvh_mp()


class TestIOHandler(TestCase):
    def test_header(self):
        self.fail()

    def test_dvh_data(self):
        obj = IOHandler(dvh_data)
        assert obj.dvh_data

    def test_to_dvh_file(self):
        # saving dvh file
        file_path = os.path.join(DATA_DIR, "test_dvh.dvh")
        obj = IOHandler(dvh_data)
        obj.to_dvh_file(file_path)

        # def test_read_dvh_file(self):

        obj = IOHandler(dvh_data)
        f_dvh_dict = obj.read_dvh_file(file_path)
        self.assertDictEqual(f_dvh_dict, dvh_data)

    def test_to_json_file(self):
        file_path = os.path.join(DATA_DIR, "test_json_dvh.jdvh")
        obj = IOHandler(dvh_data)
        obj.to_json_file(file_path)

    def test_read_json_file(self):
        # try to test function
        # TODO debug slow reading in pytests
        file_path = os.path.join(DATA_DIR, "test_json_dvh.jdvh")
        obj = IOHandler(dvh_data)
        j_dvh_dict = obj.read_json_file(file_path)
        self.assertDictEqual(j_dvh_dict, dvh_data)
