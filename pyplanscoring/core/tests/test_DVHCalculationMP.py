from unittest import TestCase

from core.calculation import PyStructure, DVHCalculationMP
from core.tests import ptv70, brain, body, lens, dose_3d, spinal_cord

grid_up = (0.2, 0.2, 0.2)
structures_dicom = [lens, body, brain, ptv70, spinal_cord]
structures_py = [PyStructure(s) for s in structures_dicom]
grids = [grid_up, None, None, None, grid_up]


class TestDVHCalculationMP(TestCase):
    def test_calc_data(self):
        # test not giving a list of structures
        with self.assertRaises(TypeError):
            calc_mp = DVHCalculationMP(dose_3d, (1, 2, 3), grids)

        # test init structures dose and grids
        with self.assertRaises(TypeError):
            calc_mp = DVHCalculationMP(dose_3d, structures_dicom, grids)

        # test init structures dose and grids
        with self.assertRaises(TypeError):
            calc_mp = DVHCalculationMP([], structures_py, grids)

        # test grid initialization same size
        with self.assertRaises(TypeError):
            calc_mp = DVHCalculationMP(dose_3d, structures_py, ())

        with self.assertRaises(TypeError):
            calc_mp = DVHCalculationMP(dose_3d, structures_py, [0.1, None])

        with self.assertRaises(TypeError):
            calc_mp = DVHCalculationMP(dose_3d, structures_py, ())

        with self.assertRaises(ValueError):
            calc_mp = DVHCalculationMP(dose_3d, structures_py, [(0.1, 2), None])
            # test give correct grids type

        # test_ adding structures and grids with different sizes
        with self.assertRaises(ValueError):
            calc_mp = DVHCalculationMP(dose_3d, structures_py, grids[:-2])

        # assert calculation data
        calc_mp = DVHCalculationMP(dose_3d, structures_py, grids)
        assert calc_mp.calc_data

    def test_calculate(self):
        # assert calculation data
        calc_mp = DVHCalculationMP(dose_3d, structures_py, grids)
        struc_dvh = calc_mp.calculate(structures_py[0], grids[0], dose_3d, True)
        assert struc_dvh

    def test_calculate_dvh_mp(self):
        calc_mp = DVHCalculationMP(dose_3d, structures_py, grids, verbose=True)
        result_mp = calc_mp.calculate_dvh_mp()
        assert result_mp
